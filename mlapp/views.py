from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from mlapp.serializers import WineSerializers
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import os
import joblib

class WineView(APIView):

	def post(self, request, *args, **kwargs):
		model_path = os.getcwd()+'\\model.joblib'
		scaler_path = os.getcwd()+'\\scaler.joblib'
		print(model_path)
		print()
		print()
		print()
		serializer = WineSerializers(data=request.data)
		serializer.is_valid()
		loaded_model = joblib.load(model_path)
		scaler = joblib.load(scaler_path)
		df = request.data
		
		print(request.data)
		print(list(map(float,list(df.values()))))
		unit = np.array(list(map(float,list(df.values()))))
		unit = unit.reshape(1, -1)
		unit_t = scaler.transform(unit)
		y_pred = loaded_model.predict(unit_t)
		return Response({'quality':y_pred})
		#return Response({"data not valid"})