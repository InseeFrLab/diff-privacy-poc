import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du dataset
df = sns.load_dataset("penguins")

st.title("Explorateur de données : Manchots 🐧")

# Choix de la variable numérique
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_var = st.selectbox("Choisissez une variable numérique :", numeric_cols)

# Calcul du quantile
quantile_alpha = st.slider("Quantile à afficher :", 0.0, 1.0, 0.5)

# Histogramme
st.subheader(f"Histogramme de {selected_var} avec le quantile {quantile_alpha:.2f}")

fig, ax = plt.subplots()
data = df[selected_var].dropna()
quantile_val = data.quantile(quantile_alpha)

sns.histplot(data, stat="percent", bins=30, color="skyblue", ax=ax)
ax.axvline(quantile_val, color='red', linestyle='--', linewidth=2, label=f'Quantile {quantile_alpha:.2f}')
ax.legend()
ax.set_xlabel(selected_var)
ax.set_ylabel("Pourcentage")
st.pyplot(fig)
