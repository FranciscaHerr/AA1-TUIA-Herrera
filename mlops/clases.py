from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# 1. Transformador para eliminar columnas 
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# 2. Transformador para filtrar ubicaciones
class LocationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, valid_locations):
        self.valid_locations = valid_locations
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X['Location'].isin(self.valid_locations)]

# 3. Transformador para componentes del viento
class WindComponentsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.direcciones = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f'Ángulo_{col}'] = X[col].map(self.direcciones) * (np.pi / 180)
            X[f'{col}_sen'] = np.sin(X[f'Ángulo_{col}'])
            X[f'{col}_cos'] = np.cos(X[f'Ángulo_{col}'])
            X.drop(columns=[f'Ángulo_{col}', col], inplace=True, errors='ignore')
        return X

# 4. Transformador para componentes de fecha
class DateComponentsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Date'] = pd.to_datetime(X['Date'])
        X['Dia_del_año'] = X['Date'].dt.dayofyear
        X['Angulo_Date'] = (X['Dia_del_año'] / 360) * 360
        radianes = np.deg2rad(X['Angulo_Date'])
        X['Date_sen'] = np.sin(radianes)
        X['Date_cos'] = np.cos(radianes)
        return X.drop(columns=['Dia_del_año', 'Angulo_Date', 'Date'], errors='ignore')
    
# 5. Transformador para eliminación de valores atípicos    
#class OutlierCapper(BaseEstimator, TransformerMixin):
#    def __init__(self, variables, quantile=0.99):
#        self.variables = variables
#        self.quantile = quantile
#        self.caps_ = {}
#        
#    def fit(self, X, y=None):
#        for var in self.variables:
#            self.caps_[var] = X[var].quantile(self.quantile)
#        return self
#    
#    def transform(self, X):
#        X = X.copy()
#        for var, cap in self.caps_.items():
#            X[var] = X[var].clip(upper=cap)
#        return X

# 6. Simple Imputer 
class DataFrameImputer(SimpleImputer):
    def transform(self, X):
        # Convertir el resultado del imputer a DataFrame y mantener nombres de columnas
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)