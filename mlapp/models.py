from django.db import models

# Create your models here.
class Wine(models.Model):
	
	fixedAcidity=models.FloatField(default=0)
	volatileAcidity = models.FloatField(default=0)
	citricAcid = models.FloatField(default=0)
	residualSugar = models.FloatField(default=0)
	chlorides = models.FloatField(default=0)
	freeSulfurDioxide = models.FloatField(default=0)
	totalSulfurDioxide = models.FloatField(default=0)
	density = models.FloatField(default=0)
	pH = models.FloatField(default=0)
	sulphates = models.FloatField(default=0)
	alcohol = models.FloatField(default=0)
	

	def __str__(self):
		return str(self.alcohol)
