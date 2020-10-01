
from rest_framework import serializers
from . models import Wine

class WineSerializers(serializers.ModelSerializer):
	class Meta:
		model = Wine
		fields='__all__'