from src.app import app
from flask import Flask
from src.db import db


# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(port=8059)
