import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st

# Configurar parámetros iniciales
optimizer_choice = "adam"
loss_regression = "mse"
loss_classification = "sparse_categorical_crossentropy"
metrics_regression = "mae"
metrics_classification = "accuracy"
epochs = 50

# Función para generar datos sintéticos de diamantes
@st.cache_data
def generate_diamond_data(n_samples=2000):
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        carat = np.random.uniform(0.2, 5.0)
        depth = np.random.uniform(55, 75)
        table = np.random.uniform(52, 67)
        x = carat * np.random.uniform(5, 7)
        y = x * np.random.uniform(0.9, 1.1)
        z = x * np.random.uniform(0.55, 0.75)
        
        # Calcular precio basado en características reales
        base_price = (carat ** 2) * 7000 + np.random.uniform(-1000, 1000)
        base_price += (x * y * z) * 50
        base_price += (70 - depth) * 100 if depth < 70 else (depth - 70) * -100
        base_price += (60 - table) * 50 if table < 60 else (table - 60) * -50
        
        price = max(base_price, 300)
        
        # Crear categorías de precio para clasificación
        if price < 2000:
            price_category = 0  # Bajo
        elif price < 8000:
            price_category = 1  # Medio
        else:
            price_category = 2  # Alto
            
        data.append([carat, depth, table, x, y, z, price, price_category])
    
    columns = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price', 'price_category']
    return pd.DataFrame(data, columns=columns)

# Cargar datos
df = generate_diamond_data()

# Eliminar valores faltantes (ya no hay, pero por consistencia)
df.dropna(inplace=True)

# Identificar valores atípicos
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Preparar datos para regresión
X_regression = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y_regression = df['price']

# Preparar datos para clasificación
X_classification = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y_classification = df['price_category']

# Normalizar características
scaler = StandardScaler()
X_regression_scaled = scaler.fit_transform(X_regression)
X_classification_scaled = scaler.transform(X_classification)

# Dividir datos
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression_scaled, y_regression, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification_scaled, y_classification, test_size=0.2, random_state=42
)

# Función para crear modelo MLP con arquitectura x,16,8,1
def create_regression_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(6,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

# Función para crear modelo MLP para clasificación
def create_classification_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(6,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 categorías
    ])
    return model

# Entrenar modelos
@st.cache_resource
def train_models():
    # Modelo de regresión
    regression_model = create_regression_model()
    regression_model.compile(
        optimizer=optimizer_choice,
        loss=loss_regression,
        metrics=[metrics_regression]
    )
    
    reg_history = regression_model.fit(
        X_train_reg, y_train_reg,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test_reg, y_test_reg),
        verbose=0
    )
    
    # Modelo de clasificación
    classification_model = create_classification_model()
    classification_model.compile(
        optimizer=optimizer_choice,
        loss=loss_classification,
        metrics=[metrics_classification]
    )
    
    clf_history = classification_model.fit(
        X_train_clf, y_train_clf,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test_clf, y_test_clf),
        verbose=0
    )
    
    return regression_model, classification_model, reg_history, clf_history

regression_model, classification_model, reg_history, clf_history = train_models()

# Realizar predicciones
y_pred_reg = regression_model.predict(X_test_reg, verbose=0)
y_pred_clf = classification_model.predict(X_test_clf, verbose=0)
y_pred_clf_classes = np.argmax(y_pred_clf, axis=1)

# Funciones de gráficos dinámicos
def plot_outliers():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[['carat', 'depth', 'table', 'x', 'y', 'z']], ax=ax)
    ax.set_title("Distribución de Características y Valores Atípicos")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_histogram():
    numeric_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Histograma de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

def plot_correlation():
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[['carat', 'depth', 'table', 'x', 'y', 'z', 'price']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, center=0)
    ax.set_title("Matriz de Correlación")
    st.pyplot(fig)

def plot_training_history():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pérdida regresión
    ax1.plot(reg_history.history['loss'], label='Entrenamiento', color='blue')
    ax1.plot(reg_history.history['val_loss'], label='Validación', color='red')
    ax1.set_title('Pérdida - Modelo Regresión')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Métrica regresión
    ax2.plot(reg_history.history['mae'], label='Entrenamiento', color='blue')
    ax2.plot(reg_history.history['val_mae'], label='Validación', color='red')
    ax2.set_title('MAE - Modelo Regresión')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    # Pérdida clasificación
    ax3.plot(clf_history.history['loss'], label='Entrenamiento', color='green')
    ax3.plot(clf_history.history['val_loss'], label='Validación', color='orange')
    ax3.set_title('Pérdida - Modelo Clasificación')
    ax3.set_xlabel('Épocas')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Métrica clasificación
    ax4.plot(clf_history.history['accuracy'], label='Entrenamiento', color='green')
    ax4.plot(clf_history.history['val_accuracy'], label='Validación', color='orange')
    ax4.set_title('Accuracy - Modelo Clasificación')
    ax4.set_xlabel('Épocas')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_price_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribución de precios
    ax1.hist(df['price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribución de Precios')
    ax1.set_xlabel('Precio ($)')
    ax1.set_ylabel('Frecuencia')
    
    # Distribución de categorías
    category_counts = df['price_category'].value_counts().sort_index()
    categories = ['Bajo (<$2000)', 'Medio ($2000-$8000)', 'Alto (>$8000)']
    ax2.bar(categories, category_counts.values, color=['lightgreen', 'yellow', 'lightcoral'])
    ax2.set_title('Distribución de Categorías de Precio')
    ax2.set_ylabel('Cantidad')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_network_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.1, 0.5, 'Input\n(6 neuronas)\ncarat, depth, table\nx, y, z', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), 
            ha='center', va='center', fontsize=10)
    
    ax.text(0.35, 0.5, 'Hidden 1\n(16 neuronas)\nReLU', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"), 
            ha='center', va='center', fontsize=10)
    
    ax.text(0.6, 0.5, 'Hidden 2\n(8 neuronas)\nReLU', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"), 
            ha='center', va='center', fontsize=10)
    
    ax.text(0.85, 0.7, 'Output Regresión\n(1 neurona)\nLinear', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), 
            ha='center', va='center', fontsize=10)
    
    ax.text(0.85, 0.3, 'Output Clasificación\n(3 neuronas)\nSoftmax', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"), 
            ha='center', va='center', fontsize=10)
    
    # Flechas
    ax.arrow(0.2, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.45, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.5, 0.08, 0.15, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.5, 0.08, -0.15, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Arquitectura MLP: 6 → 16 → 8 → 1/3')
    ax.axis('off')
    st.pyplot(fig)

# Interfaz Streamlit
st.title("🔬 Análisis MLP para Predicción de Precios de Diamantes")
st.markdown("**Multilayer Perceptron - Regresión y Clasificación**")

# Menú lateral
menu = ["Preprocesamiento de Datos", "Métricas del Modelo", "Realizar Predicciones"]
choice = st.sidebar.selectbox("Selecciona una opción", menu)

# Configuración del modelo
st.sidebar.subheader("⚙️ Configuración del Modelo")
optimizer_choice = st.sidebar.selectbox("Optimizador", ["adam", "sgd", "rmsprop"])
epochs = st.sidebar.slider("Épocas", 10, 200, 50)

# Variables de entrada
st.sidebar.subheader("📊 Variables de entrada")
feature_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
input_values = {}

for feature in feature_columns:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    
    input_values[feature] = st.sidebar.slider(
        f"{feature.capitalize()}",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=(max_val - min_val) / 100
    )

input_values_df = pd.DataFrame([input_values])
input_scaled = scaler.transform(input_values_df)

# Contenido según menú
if choice == "Preprocesamiento de Datos":
    st.subheader("📋 Datos después del preprocesamiento")
    st.write(f"**Número de muestras:** {len(df)}")
    st.write(f"**Características:** {', '.join(feature_columns)}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primeras 10 filas:**")
        st.dataframe(df.head(10))
    
    with col2:
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df[feature_columns + ['price']].describe())
    
    st.subheader("📊 Visualizaciones")
    
    plot_outliers()
    plot_histogram()
    plot_correlation()
    plot_price_distribution()
    plot_network_architecture()

elif choice == "Métricas del Modelo":
    st.subheader("🎯 Resultados del Modelo de Regresión")
    
    # Métricas regresión
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse:.2f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("RMSE", f"{rmse:.2f}")
    
    st.text("📈 Arquitectura: 6 → 16 (ReLU) → 8 (ReLU) → 1 (Linear)")
    
    st.subheader("🏷️ Resultados del Modelo de Clasificación")
    
    # Métricas clasificación
    accuracy = accuracy_score(y_test_clf, y_pred_clf_classes)
    
    st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")
    
    st.text("Matriz de Confusión:")
    cm = confusion_matrix(y_test_clf, y_pred_clf_classes)
    st.text(str(cm))
    
    st.text("Reporte de Clasificación:")
    report = classification_report(y_test_clf, y_pred_clf_classes, 
                                 target_names=['Bajo', 'Medio', 'Alto'])
    st.text(report)
    
    st.text("📈 Arquitectura: 6 → 16 (ReLU) → 8 (ReLU) → 3 (Softmax)")
    
    st.subheader("📊 Historial de Entrenamiento")
    plot_training_history()
    
    # Comparación de resultados
    st.subheader("🔍 Comparación de Resultados")
    st.write("""
    **Modelo de Regresión:**
    - Predice el precio exacto del diamante
    - Utiliza activación lineal en la salida
    - Métricas: MSE, MAE, RMSE
    
    **Modelo de Clasificación:**
    - Clasifica en 3 categorías de precio
    - Utiliza activación softmax en la salida
    - Métricas: Accuracy, Precision, Recall, F1-Score
    
    **Conclusiones:**
    - Ambos modelos comparten la misma arquitectura base (6→16→8)
    - La diferencia principal está en la capa de salida y función de activación
    - El modelo de regresión proporciona valores continuos
    - El modelo de clasificación facilita la interpretación categórica
    """)

elif choice == "Realizar Predicciones":
    st.subheader("🔮 Predicciones con los valores seleccionados")
    st.dataframe(input_values_df)
    
    # Predicción regresión
    reg_pred = regression_model.predict(input_scaled, verbose=0)[0][0]
    
    # Predicción clasificación
    clf_pred_proba = classification_model.predict(input_scaled, verbose=0)[0]
    clf_pred_class = np.argmax(clf_pred_proba)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Predicción de Precio (Regresión)")
        st.success(f"**Precio Predicho: ${reg_pred:,.2f}**")
        
        # Mostrar arquitectura
        st.info("🏗️ Arquitectura: 6 → 16 → 8 → 1 (Linear)")
    
    with col2:
        st.subheader("🏷️ Predicción de Categoría (Clasificación)")
        categories = ['Bajo (<$2,000)', 'Medio ($2,000-$8,000)', 'Alto (>$8,000)']
        category_colors = ['🟢', '🟡', '🔴']
        
        predicted_category = categories[clf_pred_class]
        confidence = clf_pred_proba[clf_pred_class] * 100
        
        st.success(f"**{category_colors[clf_pred_class]} Categoría: {predicted_category}**")
        st.info(f"**Confianza: {confidence:.1f}%**")
        
        # Mostrar probabilidades
        st.write("**Probabilidades por categoría:**")
        for i, (cat, prob) in enumerate(zip(categories, clf_pred_proba)):
            st.write(f"{category_colors[i]} {cat}: {prob*100:.1f}%")
        
        st.info("🏗️ Arquitectura: 6 → 16 → 8 → 3 (Softmax)")
    
    # Información adicional
    st.subheader("ℹ️ Información del Diamante")
    st.write(f"""
    - **Volumen aproximado:** {input_values['x'] * input_values['y'] * input_values['z']:.2f} mm³
    - **Ratio profundidad:** {input_values['depth']:.1f}%
    - **Ratio mesa:** {input_values['table']:.1f}%
    - **Peso por volumen:** {input_values['carat'] / (input_values['x'] * input_values['y'] * input_values['z']) * 1000:.2f} quilates/cm³
    """)

# Información del pie de página
st.markdown("---")
st.markdown("**📚 Proyecto MLP - Arquitectura: X → 16 → 8 → 1**")
st.markdown("*Datos normalizados | División 80/20 | Funciones de activación: ReLU + Linear/Softmax*")
