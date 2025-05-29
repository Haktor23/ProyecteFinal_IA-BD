from predictor_ozono import predecir_ozono

valores_entrada = {
    'co': 0.15,
    'so2': 4.1,
    'pm10': 35,
    'pm25': 20,
   
}
resultado = predecir_ozono(valores_entrada, modelo='xgboost')
print(f"🌿 Predicción de ozono: {resultado:.2f}")
#pip install xgboost
#pip install sklearn