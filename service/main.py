import joblib
from fastapi import FastAPI

from child_model import Child

app = FastAPI()

model0 = joblib.load('./atencion_prevencion_rf_mode.pkl')
sexo = joblib.load('./sexo_parent.pkl')
prueba_aplicada = joblib.load('./prueba_aplicada.pkl')


@app.post("/predict")
async def root(child: Child):
    data = dict(child)
    data['sexo_parent'] = sexo.transform([child.sexo_parent.value])[0]
    data['sexo_child'] = sexo.transform([child.sexo_parent.value])[0]
    data['prueba_aplicada'] = prueba_aplicada.transform([child.prueba_aplicada.value])[0]
    prediction = model0.predict([
        list(data.values())
    ])

    return {"message": "Se ejecutó correctamente el modelo", "state": prediction}
