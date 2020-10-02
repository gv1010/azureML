from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from mlapp.serializers import WineSerializers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from rest_framework import status
import numpy as np
import pandas as pd
import os
import joblib
from mlapp.models import Wine

class WineView(APIView):
	def get(self, request, *args, **kwargs):
		return Response(["Success"])
		
	def post(self, request, *args, **kwargs):
		model_path = os.getcwd()+'/model.joblib'
		scaler_path = os.getcwd()+'/scaler.joblib'
		print()
		#serializer = WineSerializers(data=request.data)
		#serializer.is_valid()
		loaded_model = joblib.load(model_path)
		scaler = joblib.load(scaler_path)
		df = request.data
		unit = np.array(list(map(float,list(df.values()))))
		unit = unit.reshape(1, -1)
		print()
		print(unit)
		unit_t = scaler.transform(unit)
		print()
		print(unit_t)
		y_pred = loaded_model.predict(unit_t)
		print("Predicted Wine Quality:", y_pred)
		return Response([y_pred], status=status.HTTP_200_OK)
