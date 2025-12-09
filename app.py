import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# ============================
# LOAD TFLITE MODEL
# ============================

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def tflite_predict(input_array):
    interpreter.set_tensor(input_details[0]["index"], input_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


# Load preprocessor
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# ============================
#   STREAMLIT UI
# ============================
st.title("🔁 Bi-Directional FDM Property Predictor — PSO Optimizer")

mode = st.radio("Select Mode:", ["Prediction Mode", "Optimization Mode"])

# =====================================================================
# ======================= PREDICTION MODE ==============================
# =====================================================================

if mode == "Prediction Mode":
    st.subheader("Input Process Parameters")

    material = st.selectbox("Material", ["PLA", "ABS"])
    layer_height = st.number_input("Layer Height (mm)", min_value=0.05, max_value=1.0, value=0.2, step=0.01)
    wall_thickness = st.number_input("Wall Thickness (mm)", min_value=0.4, max_value=3.0, value=1.2, step=0.01)
    infill_density = st.number_input("Infill Density (%)", min_value=0, max_value=100, value=50, step=1)
    infill_pattern = st.selectbox("Infill Pattern", ["Grid", "Honeycomb"])
    nozzle_temperature = st.number_input("Nozzle Temperature (°C)", min_value=180, max_value=250, value=210)
    bed_temperature = st.number_input("Bed Temperature (°C)", min_value=20, max_value=120, value=60)
    print_speed = st.number_input("Print Speed (mm/s)", min_value=10, max_value=120, value=50)
    fan_speed = st.number_input("Fan Speed (%)", min_value=0, max_value=100, value=100)
    roughness = st.number_input("Roughness (µm)", min_value=0.0, max_value=1.0, value=0.2, step=0.001)

    if st.button("🔮 Predict Properties"):

        input_df = pd.DataFrame([{
            'material': material,
            'layer_height': layer_height,
            'wall_thickness': wall_thickness,
            'infill_density': infill_density,
            'infill_pattern': infill_pattern,
            'nozzle_temperature': nozzle_temperature,
            'bed_temperature': bed_temperature,
            'print_speed': print_speed,
            'fan_speed': fan_speed,
            'roughness': roughness
        }])

        input_data = preprocessor.transform(input_df)

        preds = tflite_predict(input_data)

        st.success(f"Tensile Strength: {preds[0][0]:.3f} MPa")
        st.success(f"Elongation: {preds[0][1]:.3f} %")


# =====================================================================
# ===================== OPTIMIZATION MODE =============================
# =====================================================================

else:
    st.subheader("Input Target Properties")

    tensile_target = st.number_input("🎯 Target Tensile Strength (MPa)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
    elongation_target = st.number_input("🎯 Target Elongation (%)", min_value=0.0, max_value=200.0, value=5.0, step=0.1)

    # ---- Advanced PSO Options ----
    with st.expander("⚙️ Advanced PSO Options (Optional)"):
        n_particles = st.slider("PSO Particles", 8, 200, 30)
        n_iter = st.slider("PSO Iterations", 10, 1000, 200)
        inertia = st.number_input("Inertia (w)", 0.0, 1.5, 0.7)
        cognitive = st.number_input("Cognitive (c1)", 0.0, 2.5, 1.5)
        social = st.number_input("Social (c2)", 0.0, 2.5, 1.5)
        early_stop_tol = st.number_input("Stop if MSE < ", min_value=0.0, value=1e-5, step=1e-6, format="%.8f")

    # Defaults (if expander not touched)
    if "n_particles" not in locals():
        n_particles = 30
        n_iter = 200
        inertia = 0.7
        cognitive = 1.5
        social = 1.5
        early_stop_tol = 1e-5

    if st.button("🚀 Optimize Using PSO"):
        st.info("Running PSO Optimization...")
        progress_bar = st.progress(0)
        status = st.empty()

        bounds = np.array([
            [0.05, 1.0],
            [0.4, 3.0],
            [0.0, 100.0],
            [180.0, 250.0],
            [20.0, 120.0],
            [10.0, 120.0],
            [0.0, 100.0],
            [0.0, 1.0]
        ])

        dim = bounds.shape[0]
        rng = np.random.default_rng(int(time.time()))

        X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, dim))
        V = rng.normal(0, 0.1, size=(n_particles, dim))

        pbest = X.copy()
        pbest_scores = np.full(n_particles, np.inf)
        gbest = None
        gbest_score = np.inf
        gbest_pred = None

        # Evaluate batch
        def evaluate_batch(X_batch):

            rows = []
            for xi in X_batch:
                rows.append({
                    'material': 'PLA',
                    'layer_height': xi[0],
                    'wall_thickness': xi[1],
                    'infill_density': xi[2],
                    'infill_pattern': 'Grid',
                    'nozzle_temperature': xi[3],
                    'bed_temperature': xi[4],
                    'print_speed': xi[5],
                    'fan_speed': xi[6],
                    'roughness': xi[7]
                })

            df = pd.DataFrame(rows)
            inp = preprocessor.transform(df)
            preds = tflite_predict(inp)

            target = np.array([tensile_target, elongation_target])
            scores = np.sum((preds - target) ** 2, axis=1)

            return preds, scores

        # Initial
        _, scores = evaluate_batch(X)
        pbest_scores = scores.copy()

        idx = np.argmin(scores)
        gbest = X[idx].copy()
        gbest_score = scores[idx]
        gbest_pred, _ = evaluate_batch(np.array([gbest]))

        # PSO Loop
        for it in range(n_iter):

            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))

            V = inertia * V + cognitive * r1 * (pbest - X) + social * r2 * (gbest - X)
            X = X + V
            X = np.clip(X, bounds[:, 0], bounds[:, 1])

            preds, scores = evaluate_batch(X)

            improved = scores < pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest[improved] = X[improved]

            idx = np.argmin(pbest_scores)
            if pbest_scores[idx] < gbest_score:
                gbest_score = pbest_scores[idx]
                gbest = pbest[idx].copy()
                gbest_pred, _ = evaluate_batch(np.array([gbest]))

            progress_bar.progress((it + 1) / n_iter)
            status.text(f"Iter {it+1}/{n_iter} — Best MSE: {gbest_score:.6f}")

            if gbest_score < early_stop_tol:
                break

        best_params = {
            "Material": "PLA",
            "Layer Height": round(float(gbest[0]), 3),
            "Wall Thickness": round(float(gbest[1]), 3),
            "Infill Density": round(float(gbest[2]), 1),
            "Infill Pattern": "Grid",
            "Nozzle Temperature": round(float(gbest[3]), 1),
            "Bed Temperature": round(float(gbest[4]), 1),
            "Print Speed": round(float(gbest[5]), 1),
            "Fan Speed": round(float(gbest[6]), 1),
            "Roughness": round(float(gbest[7]), 3),
        }

        st.success("Optimization Complete!")
        st.json(best_params)
        st.write(f"Predicted Tensile = {gbest_pred[0][0]:.3f} MPa")
        st.write(f"Predicted Elongation = {gbest_pred[0][1]:.3f} %")
