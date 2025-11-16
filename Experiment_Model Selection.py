import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Activation
from tensorflow.keras.regularizers import l2
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import os
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.initializers import GlorotUniform
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.layers import Input, Dense, Activation, Add, BatchNormalization, LeakyReLU
from scipy.spatial.distance import mahalanobis
import tensorflow.keras.backend as K
import multiprocessing

num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cores)

tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
tf.config.set_soft_device_placement(True)


df= pd.read_csv("dataset.csv",index_col=0)
df = df[df['MOF_ID'] != 'hMOF-1002818']

metals=['chem_metal_Sc', 'chem_metal_Ti',
       'chem_metal_V', 'chem_metal_Cr', 'chem_metal_Mn', 'chem_metal_Fe',
       'chem_metal_Co', 'chem_metal_Ni', 'chem_metal_Cu', 'chem_metal_Zn',
       'chem_metal_Y', 'chem_metal_Zr', 'chem_metal_Nb', 'chem_metal_Mo',
       'chem_metal_Tc', 'chem_metal_Ru', 'chem_metal_Rh', 'chem_metal_Pd',
       'chem_metal_Ag', 'chem_metal_Cd', 'chem_metal_Hf', 'chem_metal_Ta',
       'chem_metal_W', 'chem_metal_Re', 'chem_metal_Os', 'chem_metal_Ir',
       'chem_metal_Pt', 'chem_metal_Au', 'chem_metal_Hg', 'chem_metal_Al',
       'chem_metal_Ga', 'chem_metal_In', 'chem_metal_Sn', 'chem_metal_Pb',
       'chem_metal_Bi', 'chem_metal_La', 'chem_metal_Ce', 'chem_metal_Pr',
       'chem_metal_Nd', 'chem_metal_Sm', 'chem_metal_Eu', 'chem_metal_Gd',
       'chem_metal_Tb', 'chem_metal_Dy', 'chem_metal_Ho', 'chem_metal_Er',
       'chem_metal_Tm', 'chem_metal_Yb', 'chem_metal_Lu',]
nonmetals=['chem_num_atoms',
 'chem_volume',
 'chem_density',
 'chem_avg_atomic_mass',
 'chem_avg_electronegativity',
 'chem_electronegativity_variance',
 'chem_metal_fraction',
 'chem_num_unique_elements',
 'chem_metal_atom_count',
 'chem_volume_per_atom',
 'geo_surface_area_m2g',
 'geo_surface_area_m2cm3',
 'geo_void_fraction',
 'geo_pld',
 'geo_lcd',
 'link_linker_atom_fraction',
 'link_linker_bond_length_mean',
 'link_linker_bond_length_std',
 'link_metal_coord_number_mean',
 'topo_avg_node_connectivity',
 'topo_avg_ring_size',
 'topo_coordination_number_mean',
 'topo_degree_assortativity',
 'topo_degree_centrality_mean',
 'topo_graph_density',
 'topo_graph_entropy',
 'topo_graph_transitivity',
 'topo_largest_cc_fraction',
 'topo_node_connectivity_std',
 'topo_num_connected_components',
 'topo_num_edges',
 'topo_num_nodes']
features=nonmetals+metals

df_cleaned = df.dropna()
df_cleaned = df_cleaned.reset_index()
print("NaNs in X_scaled:", df_cleaned.isnull().sum().sum())
color_normal = "b"    # strong cobalt blue
color_anomaly = "r"   # vivid red (high contrast)

m=3

params = {
    # Font family
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],

    # Font sizes
    "axes.labelsize": 10*m,
    "font.size": 10*m,
    "legend.fontsize": 10*m,
    "xtick.labelsize": 9*m,
    "ytick.labelsize": 9*m,

    # Style for axis labels (xlabel, ylabel)
    'axes.labelweight': 'bold',
    'axes.labelcolor': 'black',

    # General styles for other elements
    'font.weight': 'bold',       # Makes title, etc., bold
    'xtick.color': 'black',      # Sets tick label color
    'ytick.color': 'black',
    'legend.labelcolor': 'black'
}

plt.rcParams.update(params)

RANDOM_SEED = 0
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

color_anomaly = 'red'
color_normal = 'blue'
input_dim = 81
latent_dims = [64,32,16,8,4,2,1]
step_sizes = [64,32,16,8,4,2,1]


bs = 32

act=Activation('relu')
ep = 200 # Increased epochs for meaningful training

kernel_init=None

l2_strength = 0.001
reg = l2(l2_strength)

reg = reg
lr=1e-3
patience=10

output_dir='Experiment Model Selection'
for random_state in [1,2,]:
    K.clear_session()
    results_list = []
    X_train_val_full, X_test_full = train_test_split(df_cleaned, test_size=0.2, random_state=random_state)
    X_train_full, X_val_full = train_test_split(X_train_val_full, test_size=0.25, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_full[nonmetals])
    
    X_train_nonmetal_np = scaler_X.transform(X_train_full[nonmetals])
    X_train_nonmetal = pd.DataFrame(X_train_nonmetal_np, index=X_train_full.index, columns=nonmetals)
    X_train = pd.concat([X_train_nonmetal, X_train_full[metals]], axis=1)
    
    X_val_nonmetal_np = scaler_X.transform(X_val_full[nonmetals])
    X_val_nonmetal = pd.DataFrame(X_val_nonmetal_np, index=X_val_full.index, columns=nonmetals)
    X_val = pd.concat([X_val_nonmetal, X_val_full[metals]], axis=1)
    
    X_test_nonmetal_np = scaler_X.transform(X_test_full[nonmetals])
    X_test_nonmetal = pd.DataFrame(X_test_nonmetal_np, index=X_test_full.index, columns=nonmetals)
    X_test = pd.concat([X_test_nonmetal, X_test_full[metals]], axis=1)
    
    X_full_dataset=pd.concat([X_train, X_val,X_test], axis=0)
    for latent_dim in latent_dims:
        for step_size in step_sizes:
    
            print(f"\n===== Training STABILIZED Model: Latent Dim = {latent_dim}, Step Size = {step_size} =====\n")
            
            encoder_neurons = list(range(input_dim - step_size, latent_dim, -step_size))
            if not encoder_neurons:
                print('No hidden layers')
                continue
            decoder_neurons = encoder_neurons[::-1]
            
            # --- Build Model ---
            input_layer = Input(shape=(input_dim,))
            x = input_layer
            
            # --- ENCODER ---
            # FIX: A simple, standard encoder is more stable than a flawed residual one.
            for neurons in encoder_neurons:
                x = Dense(neurons)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            
            # --- LATENT SPACE ---
            # FIX: The latent space must be linear (no activation function).
            encoded = Dense(latent_dim, name='latent_space')(x)
            x = encoded # Pass the linear latent space directly to the decoder
            
            # --- DECODER ---
            for neurons in decoder_neurons:
                x = Dense(neurons)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            
            # --- OUTPUT ---
            output_layer = Dense(input_dim, activation='linear')(x)
            autoencoder = Model(inputs=input_layer, outputs=output_layer)
            
            # --- Compile and Train ---
            optimizer = Adam(learning_rate=lr)
            autoencoder.compile(optimizer=optimizer, loss='mse')
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, min_lr=1e-5, verbose=0)
    
    
            
            train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, X_train.values)).batch(bs).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val.values, X_val.values)).batch(bs).prefetch(tf.data.AUTOTUNE)
    
    
            for _ in range(3):
                _ = autoencoder.train_on_batch(X_train[:bs], X_train[:bs])
            
            history = autoencoder.fit(
                train_ds,
                epochs=ep,
                validation_data=val_ds,
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )  
            pd.DataFrame(history.history).to_csv(f'{output_dir}/{random_state}/hist/hist_{latent_dim}_{step_size}.csv')
    
            ########################################
            reconstructions_train = autoencoder.predict(X_train.values)
            errors_train = np.mean(np.square(X_train.values - reconstructions_train), axis=1)
            sorted_scores = np.sort(errors_train)
            x_elbow = np.arange(len(sorted_scores))
            y_elbow = sorted_scores
            p1, p2 = np.array([x_elbow[0], y_elbow[0]]), np.array([x_elbow[-1], y_elbow[-1]])
            dist_to_line = np.abs(np.cross(p2 - p1, p1 - np.array([x_elbow, y_elbow]).T)) / np.linalg.norm(p2 - p1)
            anomaly_threshold = y_elbow[np.argmax(dist_to_line)]
            print(f"Calculated Anomaly Threshold: {anomaly_threshold:.4f}")
    
            
            np.save(f'{output_dir}/{random_state}/numbers/anomaly_threshold_{latent_dim}_{step_size}.npy',anomaly_threshold)
            np.save(f'{output_dir}/{random_state}/numbers/errors_train_{latent_dim}_{step_size}.npy',sorted_scores)
            np.save(f'{output_dir}/{random_state}/numbers/reconstructions_train_{latent_dim}_{step_size}.npy',reconstructions_train)
            np.save(f'{output_dir}/{random_state}/numbers/X_train_{latent_dim}_{step_size}.npy',X_train.values)
            ##############################################################
            
            x = np.arange(len(sorted_scores))
            y = sorted_scores
            elbow_index = np.argmax(dist_to_line)
    
            
            figg, axx = plt.subplots(figsize=(12, 8))
            
            axx.plot(x, y, label='Sorted Anomaly Scores', color='royalblue', linewidth=4)
            
            # Plot the line from the first to the last point
            axx.plot([x[0], x[-1]], [y[0], y[-1]], 'r--', label='Reference Line', color='tomato',linewidth=4)
            
            # Highlight the elbow point
            axx.scatter(elbow_index, anomaly_threshold, color='limegreen', s=150, zorder=5, 
                        edgecolor='black', label='Elbow Point (Threshold)')
            
            # Add annotation lines for the threshold
            axx.axhline(y=anomaly_threshold, color='grey', linestyle=':', linewidth=4)
            axx.axvline(x=elbow_index, color='grey', linestyle=':', linewidth=4)
            
            #plt.title('Elbow Method for Anomaly Detection Threshold', fontweight='bold')
            axx.set_xlabel('Data Point Index (Sorted)', color='k')
            axx.set_ylabel('Anomaly Score',color='k' )
            axx.legend()
            
            axx.text(elbow_index*0.98, anomaly_threshold*1.2, f' Threshold ≈ {anomaly_threshold:.4f}', 
                     horizontalalignment='right', verticalalignment='bottom', fontsize=m*7,color='k')
    
            figg.savefig(f'{output_dir}/{random_state}/elbow/elbow_{latent_dim}_{step_size}.png')
            plt.close(figg)
    
    
            #################################################################

    
            reconstructions_test = autoencoder.predict(X_test.values, verbose=0)
            errors_test = np.mean(np.square(X_test.values - reconstructions_test), axis=1)
            predicted_labels = (errors_test > anomaly_threshold)
            test_error=autoencoder.evaluate(X_test,X_test)
            print(test_error)

            np.save(f'{output_dir}/{random_state}/numbers/errors_test_{latent_dim}_{step_size}.npy',errors_test)
            np.save(f'{output_dir}/{random_state}/numbers/reconstructions_test_{latent_dim}_{step_size}.npy',reconstructions_test)
            np.save(f'{output_dir}/{random_state}/numbers/X_test_{latent_dim}_{step_size}.npy',X_test.values)
    
            
            reducer_95 = PCA(0.95, random_state=RANDOM_SEED)
            reducer_95.fit(X_train[features])
            embedding_pca_95 = reducer_95.transform(X_test[features])
            n_components_95 = reducer_95.components_

    
            normal_pca_95 = embedding_pca_95[~predicted_labels]
            anomaly_pca_95 = embedding_pca_95[predicted_labels]
    
            maha_dist_ratio_95 = 0.0
            if normal_pca_95.shape[0] > 1 and anomaly_pca_95.shape[0] > 0:
                # --- Metric 1: Mahalanobis Distance Ratio (UPGRADED) ---
                centroid_95 = np.mean(normal_pca_95, axis=0)
                # Calculate covariance and its inverse for the normal data
                cov_matrix_95 = np.cov(normal_pca_95, rowvar=False)
                inv_cov_matrix_95 = np.linalg.inv(cov_matrix_95)
                
                # Calculate Mahalanobis distance for each point
                maha_dist_normal = [mahalanobis(p, centroid_95, inv_cov_matrix_95) for p in normal_pca_95]
                maha_dist_anomaly = [mahalanobis(p, centroid_95, inv_cov_matrix_95) for p in anomaly_pca_95]
                
                avg_maha_normal = np.mean(maha_dist_normal)
                avg_maha_anomaly = np.mean(maha_dist_anomaly)
                maha_dist_ratio_95 = avg_maha_anomaly / avg_maha_normal if avg_maha_normal > 0 else 0

            
            print(f"Anomaly Distance Ratio : {maha_dist_ratio_95:.2f}x")
    

            ##################################################################
            reducer_2d = PCA(0.95, random_state=RANDOM_SEED)
            reducer_2d.fit(X_train[features])
            embedding_pca_2d = reducer_2d.transform(X_test[features])

    
            normal_pca_2d = embedding_pca_2d[~predicted_labels]
            anomaly_pca_2d = embedding_pca_2d[predicted_labels]


    
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(normal_pca_2d[:, 0], normal_pca_2d[:, 1], c='blue', s=10, alpha=0.6)
            ax.scatter(anomaly_pca_2d[:, 0], anomaly_pca_2d[:, 1], c='red', s=20, alpha=0.9)
    
            ax.set_xlabel("PCA Component 1", fontweight='bold')
            ax.set_ylabel("PCA Component 2", fontweight='bold')
            legend_elements_2 = [
                Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=25),
                Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=25),
            ]
            ax.legend(handles=legend_elements_2, loc='upper center', bbox_to_anchor=(0.5, -0.16),fontsize=30,  frameon=False, ncols=2)
            plt.tight_layout(pad=2.0)
            plt.savefig(f'{output_dir}/{random_state}/PCA_{latent_dim}_{step_size}.png')
            plt.close(fig)
    
            
            ##################################################
            reducer_2d = PCA(n_components=2, random_state=RANDOM_SEED)
            embedding_pca_2d = reducer_2d.fit_transform(X_full_dataset)
    
            reconstructions_full = autoencoder.predict(X_full_dataset.values, verbose=0)
            errors_full = np.mean(np.square(X_full_dataset.values - reconstructions_full), axis=1)
            predicted_labels_full = (errors_full > anomaly_threshold)
            normal_pca_2d = embedding_pca_2d[~predicted_labels_full]
            anomaly_pca_2d = embedding_pca_2d[predicted_labels_full]
    
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(normal_pca_2d[:, 0], normal_pca_2d[:, 1], c='blue', s=8, alpha=0.2)
            ax.scatter(anomaly_pca_2d[:, 0], anomaly_pca_2d[:, 1], c='red', s=10, alpha=0.9)
    
            # ... (rest of your plotting code remains the same)
            ax.set_xlabel("PCA Component 1", fontweight='bold')
            ax.set_ylabel("PCA Component 2", fontweight='bold')
    
            legend_elements_2 = [
                Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=25),
                Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=25),
            ]
            ax.legend(handles=legend_elements_2, loc='upper center', bbox_to_anchor=(0.5, -0.16),fontsize=30,  frameon=False, ncols=2)
            plt.tight_layout(pad=0.5)
            plt.savefig(f'{output_dir}/{random_state}/PCA_full_{latent_dim}_{step_size}.png')
            plt.close(fig)
            ##################################################
            g=0
            for o in predicted_labels:
                if o==True:
                    g=g+1
            print(g,g/predicted_labels.shape[0]*100)
    
            
            results_list.append({
                'latent_dim': latent_dim,
                'step_size': step_size,
                'val_loss': history.history['val_loss'][-1],
                'test_error': test_error,
                'anomaly_threshold': anomaly_threshold,
                'maha_dist_ratio_95': maha_dist_ratio_95,
                'anomaly percent': g/predicted_labels.shape[0]*100,
            })
    
    
    # --- 4. Display Final Results DataFrame ---
    print("\n\n" + "="*60)
    print("           MODEL PERFORMANCE SUMMARY")
    print("="*60)
    if results_list:
        results_df = pd.DataFrame(results_list)
        pd.set_option('display.width', 1000)
        print(results_df)
        results_df.to_csv(f'{output_dir}/{random_state}/results_{random_state}.csv', index=False)
    else:
        print("No models were trained.")
