from datetime import datetime
import io
import math
import logging
from flask import Flask, json, jsonify, request, send_file
from dotenv import load_dotenv
from ultralytics import YOLO
from src.db import db
import os
from sqlalchemy import text
from PIL import Image 
import numpy as np 
load_dotenv()

app = Flask(__name__)

ml_resources = {}

app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://Sa:WasteLocker@123@167.99.246.163:1433/Wastelocker?driver=ODBC+Driver+17+for+SQL+Server'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.before_request
def startup():
    """Initialize resources before the first request is processed"""
    if 'yolo' not in ml_resources:
        ml_resources['yolo'] = YOLO("best1024.pt")
        print("ML resources initialized")

@app.teardown_appcontext
def shutdown(exception=None):
    """Release resources after application context ends"""
    ml_resources.clear()
    print("ML resources released")


# Define a function to calculate waste volume (Q) based on the provided formula
def calculate_waste_volume(area: float, angle: float = 35) -> float:
    """
    Calculate the volume of waste in cubic meters using the given formula:
    Q = (1/3) * A * sqrt(P/pi) * tan(radians(P))
    
    :param area: Area of the waste in square meters (m²).
    :param angle: Angle of the waste pile in degrees (default is 35 degrees).
    :return: Volume of waste in cubic meters (m³).
    """
    # Calculate volume based on the formula
    volume = (1/3) * area * math.sqrt(angle / math.pi) * math.tan(math.radians(angle))
    return volume

# Define a function to calculate the waste mass in kilograms
def calculate_waste_weight(volume: float, coefficient: float) -> float:
    """
    Calculate the waste mass (in kilograms) by applying the coefficient to the volume.
    
    :param volume: Volume of waste in cubic meters (m³).
    :param coefficient: Density coefficient based on waste type (kg/m³).
    :return: Weight of waste in kilograms (kg).
    """
    return volume * coefficient

# Function to calculate the area of the waste polygon using the Shoelace formula
def calculate_polygon_area(x_coords, y_coords) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    :param x_coords: List of x coordinates of the polygon vertices.
    :param y_coords: List of y coordinates of the polygon vertices.
    :return: Area of the polygon in square meters (m²).
    """
    n = len(x_coords)
    area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] for i in range(n-1)) +
                     (x_coords[-1] * y_coords[0] - x_coords[0] * y_coords[-1]))
    return area


def save_segmentation_to_db(name, class_id, confidence, box, segments, segments_area, waste_volume, total_waste, file_path, captureDate):
    try:
        # Prepare the SQL query to insert the data into the LandFills table
        query = text("""
            INSERT INTO LandFills (Name, Class, Confidence, Box_x1, Box_y1, Box_x2, Box_y2, SegmentsX, SegmentsY, SegmentsArea, WasteVolume, TotalWaste, ImagePath, CaptureDate)
            VALUES (:name, :class, :confidence, :x1, :y1, :x2, :y2, :segmentsx, :segmentsy, :segmentsarea, :wastevolume, :totalwaste, :filepath, :captureDate)
        """)

        # Execute the query
        with db.engine.connect() as connection:
            trans = connection.begin()  # Start a transaction
            try:
                connection.execute(query, {
                    'name': name,
                    'class': class_id,
                    'confidence': confidence,
                    'x1': box['x1'],
                    'y1': box['y1'],
                    'x2': box['x2'],
                    'y2': box['y2'],
                    'segmentsx': json.dumps(segments['x']),  # Store the X coordinates as a JSON string
                    'segmentsy': json.dumps(segments['y']),  # Store the Y coordinates as a JSON string
                    'segmentsarea': segments_area,  # Area of the segmentation
                    'wastevolume': waste_volume,  # Waste volume
                    'totalwaste': total_waste,  # Total adjusted waste
                    'filepath': file_path,
                    'captureDate': captureDate
                })
                trans.commit()  # Commit the transaction after the query executes successfully
                print("Segmentation result saved to the database successfully.")
            except Exception as e:
                trans.rollback()  # Rollback the transaction in case of error
                print(f"Error executing query and saving segmentation result to database: {str(e)}")

    except Exception as e:
        print(f"Error establishing database connection or executing the transaction: {str(e)}")


# Route to process image and perform YOLO predictions
@app.route("/look_for_landfills/", methods=['POST'])
def process_image():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))  # Ensure you're using PIL's Image module
    original_filename = file.filename
    file_name, file_extension = os.path.splitext(original_filename)
    current_datetime = datetime.now();
    datetimeString = current_datetime.strftime("%Y-%m-%d_%H-%M-%S");

    new_filename = f"{file_name}_{datetimeString}{file_extension}"
    folder_path = "img"
    file_path = os.path.join(folder_path, new_filename)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image.save(file_path)

    # Run the YOLO model prediction
    results = ml_resources['yolo'].predict(image, conf=0.18)
    print(results[0].tojson(normalize=False, decimals=5))
    # Extract the necessary information from the results
    for result in results:
        if result.boxes is not None:  # Check if any boxes were detected
            for box in result.boxes:
                # For each detected object, extract the data
                class_id = int(box.cls[0])  # Get the class ID
                name = result.names[class_id]  # Get the class name using the class ID
                confidence = float(box.conf[0])  # Confidence score
                box_coords = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)

                # Extract the segments (if segmentation exists)
                if result.masks is not None:
                    segments = {
                        'x': result.masks.xy[0][:, 0].tolist(),
                        'y': result.masks.xy[0][:, 1].tolist()
                    }
                else:
                    segments = {"x": [], "y": []}

                # Calculate area and other waste-related metrics
                area = calculate_polygon_area(segments["x"], segments["y"]) if segments["x"] else 0
                waste_volume = calculate_waste_volume(area)
                waste_type_coefficients = result.names
                print('waste_type_coefficients',waste_type_coefficients)
                coefficient = waste_type_coefficients.get(name, 1.0)  # Default to 1.0 if not found
                total_waste = calculate_waste_weight(waste_volume, coefficient)

                # Save this detection result to the database
                save_segmentation_to_db(
                    name=name,
                    class_id=class_id,
                    confidence=confidence,
                    box={'x1': box_coords[0], 'y1': box_coords[1], 'x2': box_coords[2], 'y2': box_coords[3]},
                    segments=segments,
                    segments_area=area,
                    waste_volume=waste_volume,
                    total_waste=total_waste,
                    file_path=file_path,
                    captureDate=current_datetime
                )

    segmentation_only = results[0].plot(boxes=False)

    # Convert the numpy array to a PIL image
    segmentation_image = Image.fromarray(segmentation_only.astype(np.uint8))

    # Convert the image to bytes and send it back as a response
    img_byte_arr = io.BytesIO()
    segmentation_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png'), 200

@app.route('/get_landfills', methods=['GET'])
def get_landfills():
    try:
        # Prepare the SQL query to fetch data from the LandFills table
        query = text("SELECT * FROM LandFills")
        
        # Execute the query and fetch all rows
        with db.engine.connect() as connection:
            result = connection.execute(query)
            rows = result.fetchall()  # Fetch all rows as tuples
        
        # Prepare the response in the desired format
        landfills = []
        for row in rows:
            # Check if SegmentsX and SegmentsY are valid strings, if not, use empty lists
            segments_x = json.loads(row[8]) if isinstance(row[8], str) else []
            segments_y = json.loads(row[9]) if isinstance(row[9], str) else []

            # Combine the coordinates into a list of tuples
            coordinates = list(zip(segments_x, segments_y))

            landfill = {
                "id": row[0],  # Assuming Id is the first column
                "idOfImage": row[2],  # Assuming IdOfImage is the second column
                "coordinates": coordinates,  # Combine SegmentsX and SegmentsY into coordinates
                "confidence": row[3],  # Assuming Confidence is the third column
                "wasteVolumeInCubicMeters": row[11],  # Assuming WasteVolume is the column at index 10
                "wasteWeightInKilograms": row[12],  # Assuming TotalWaste is at index 11
                "wasteType": row[1]  # Assuming Name is the waste type
            }
            landfills.append(landfill)

        # Return the fetched data as JSON
        return jsonify(landfills), 200  # 200 OK status code

    except Exception as e:
        print(f"Error retrieving data from the database: {str(e)}")
        return jsonify({"error": "An error occurred while fetching the data."}), 500  # 500 Internal Server Error

@app.route('/get_landfillswithparameter', methods=['GET'])
def get_landfillswithparamete():
    waste_type = request.args.get('waste_type')  # Get the waste_type from the query parameters

    try:
        # Prepare the SQL query to fetch data from the LandFills table with an optional filter
        if waste_type:
            query = text("SELECT * FROM LandFills WHERE Name = :waste_type")
            params = {'waste_type': waste_type}
        else:
            query = text("SELECT * FROM LandFills")
            params = {}

        # Execute the query and fetch all rows
        with db.engine.connect() as connection:
            result = connection.execute(query, params)
            rows = result.fetchall()
            columns = result.keys()
            landfills = [dict(zip(columns, row)) for row in rows]

        # Return the fetched data as JSON
        return jsonify(landfills), 200

    except Exception as e:
        print(f"Error retrieving data from the database: {str(e)}")
        return jsonify({"error": "An error occurred while fetching the data."}), 500



@app.route('/')
def testing():
  return "hello"

@app.route('/getdata', methods=['GET'])
def getdata():
  try:
    # Use the engine to get a connection
    with db.engine.connect() as connection:
        # Raw SQL query
        sql_query = text('SELECT * FROM LandFills')

        # Execute the query
        result = connection.execute(sql_query)

        # Fetch all records and convert them to a list of dictionaries
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in result.fetchall()]

    return jsonify(data), 200
  except Exception as e:
    return str(e), 500
