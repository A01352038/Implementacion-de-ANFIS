from anfis_model import ANFIS
import membershipfunction
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Definir la función para calcular el RMSE
def get_RMSE(real_values_list, predicted_value_list):
    return np.sqrt(np.square(np.subtract(real_values_list, predicted_value_list)).mean())

# Cargar datos
ts = pd.read_csv("hp_train.csv").sort_values(by=['SalePrice'])
X = ts[['BedroomAbvGr', 'OverallQual', 'OverallCond']].to_numpy()  # 3 Inputs
Y = ts['SalePrice'].to_numpy()  # Output

# Escalar los datos
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# Parámetros de funciones de pertenencia
mf_params = {
    'gaussmf': [{'mean': 0, 'sigma': 1}, {'mean': 0.5, 'sigma': 0.5}, {'mean': 1, 'sigma': 0.25}],
    'gbellmf': [{'a': 2, 'b': 4, 'c': 0}, {'a': 2, 'b': 4, 'c': 0.5}, {'a': 2, 'b': 4, 'c': 1}],
    'sigmf': [{'b': 1, 'c': 0}, {'b': 1, 'c': 0.5}, {'b': 1, 'c': 1}]
}

# Configuraciones
num_sets = [2, 3]  # Reducido para acelerar la experimentación inicial
learning_rates = [0.01, 0.1]
mf_types = ['gaussmf', 'gbellmf']

# Almacenar resultados
results = []

# Realizar experimentos
for num_set in num_sets:
    for lr in learning_rates:
        for mf_type in mf_types:
            mf = []
            for i in range(X_scaled.shape[1]):
                mf.append([(mf_type, mf_params[mf_type][j % len(mf_params[mf_type])]) for j in range(num_set)])

            # Crear el objeto de funciones de pertenencia
            mfc = membershipfunction.MemFuncs(mf)

            # Crear el objeto ANFIS
            anf = ANFIS(X_scaled, Y_scaled, mfc)

            # Plot the MFs pre-training
            anf.plotMF(X_scaled[:, 0], 0)
            anf.plotMF(X_scaled[:, 1], 1)
            anf.plotMF(X_scaled[:, 2], 2)

            # Entrenar el modelo
            anf.trainHybridJangOffLine(epochs=20, k=lr)

            # Calcular el RMSE
            rmse = get_RMSE(anf.Y, anf.fittedValues)
            print(f'num_set: {num_set}, learning_rate: {lr}, mf_type: {mf_type}, RMSE: {rmse}')

            # Almacenar el resultado
            results.append({'num_set': num_set, 'learning_rate': lr, 'mf_type': mf_type, 'rmse': rmse})

            # Plotting Model performance
            anf.plotErrors()
            anf.plotResults()

            # Plot the MFs post-training
            anf.plotMF(X_scaled[:, 0], 0)
            anf.plotMF(X_scaled[:, 1], 1)
            anf.plotMF(X_scaled[:, 2], 2)

# Convertir resultados a DataFrame y guardar en un archivo CSV
results_df = pd.DataFrame(results)
results_df.to_csv('anfis_experiments_results.csv', index=False)

# Mostrar la tabla de resultados
print(results_df)

# Identificar la mejor configuración
best_result = results_df.loc[results_df['rmse'].idxmin()]
print("Mejor configuración:")
print(best_result)

# Graficar los valores reales vs. las salidas del modelo para la mejor configuración
best_mf = []
for i in range(X_scaled.shape[1]):
    mf_list = [(best_result['mf_type'], mf_params[best_result['mf_type']][j % len(mf_params[best_result['mf_type']])]) for j in range(int(best_result['num_set']))]
    best_mf.append(mf_list)

best_mfc = membershipfunction.MemFuncs(best_mf)
best_anf = ANFIS(X_scaled, Y_scaled, best_mfc)
best_anf.trainHybridJangOffLine(epochs=20, k=best_result['learning_rate'])

# Plotting Model performance for the best configuration
best_anf.plotErrors()
best_anf.plotResults()

# Plot the MFs post-training for the best configuration
best_anf.plotMF(X_scaled[:, 0], 0)
best_anf.plotMF(X_scaled[:, 1], 1)
best_anf.plotMF(X_scaled[:, 2], 2)
