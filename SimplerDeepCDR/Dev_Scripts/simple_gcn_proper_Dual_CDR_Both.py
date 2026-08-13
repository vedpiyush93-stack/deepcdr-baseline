import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import numpy as np

drugs_cell_lines_ic50_df = pd.read_csv("..//data//drugs_cell_lines_ic50.csv")

pubchem_drugs_smiles_df = pd.read_csv('..//data//drugs_smile_strings.csv')

drugs_smiles_cell_lines_ic50_df = pd.merge(drugs_cell_lines_ic50_df, pubchem_drugs_smiles_df, 
                                             on = "drug_id")

drugs_smiles_cell_lines_ic50_df = drugs_smiles_cell_lines_ic50_df[["drug_id", "Cancer_Cell_Line", "Smiles", "IC50"]]

drugs_smiles_cell_lines_ic50_df["drug_id"] = drugs_smiles_cell_lines_ic50_df["drug_id"].astype(str)

import pickle

with open("..//data//drug_gcn_features.pickle", "rb") as f:
    dict_features = pickle.load(f)
    
with open("..//data//drug_gcn_normalized_adj_mats.pickle", "rb") as f:
    dict_normalized_adj_mats = pickle.load(f)
    
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(drugs_smiles_cell_lines_ic50_df.drop(["IC50"],1), drugs_smiles_cell_lines_ic50_df["IC50"].values, 
                                                     test_size = 0.20, random_state = 42)

train_gcn_feats = []
train_adj_list = []
for drug_id in x_train["drug_id"].values:
    train_gcn_feats.append(dict_features[drug_id])
    train_adj_list.append(dict_normalized_adj_mats[drug_id])
    
valid_gcn_feats = []
valid_adj_list = []
for drug_id in x_valid["drug_id"].values:
    valid_gcn_feats.append(dict_features[drug_id])
    valid_adj_list.append(dict_normalized_adj_mats[drug_id])
    
train_gcn_feats = np.array(train_gcn_feats).astype("float16")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float16")

train_adj_list = np.array(train_adj_list).astype("float16")
valid_adj_list = np.array(valid_adj_list).astype("float16")

# load models
# omic models
cancer_copy_number_model = tf.keras.models.load_model("..//Models//cancer_copy_number_model_no_norm_common")
cancer_cell_gen_expr_model = tf.keras.models.load_model("..//Models//cancer_cell_gen_expr_model_no_norm_common")
cancer_cell_gen_methy_model = tf.keras.models.load_model("..//Models//cancer_cell_gen_methy_model_no_norm")
cancer_cell_gen_mut_model = tf.keras.models.load_model("..//Models//cancer_cell_gen_mut_model_no_norm")

# load models
# drug models
pubchem_drugs_rdkit_model = tf.keras.models.load_model("..//Models//pubchem_drugs_rdkit_model_no_norm")

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

# extract drug features
drug_features_train = pubchem_drugs_rdkit_model(x_train["drug_id"].values).numpy().astype("float32")
drug_features_valid = pubchem_drugs_rdkit_model(x_valid["drug_id"].values).numpy().astype("float32")

# drug_features_train = pubchem_drugs_rdkit_model(x_train["drug_id"].values)
# drug_features_valid = pubchem_drugs_rdkit_model(x_valid["drug_id"].values)

drug_features_train = std.fit_transform(drug_features_train)

drug_features_valid = std.transform(drug_features_valid)

# extract copy number features
omics_copy_number_train = cancer_copy_number_model(x_train["Cancer_Cell_Line"].values).numpy().astype("float16")
omics_copy_number_valid = cancer_copy_number_model(x_valid["Cancer_Cell_Line"].values).numpy().astype("float16")

# extract gen expr features
omics_gen_expr_train = cancer_cell_gen_expr_model(x_train["Cancer_Cell_Line"].values).numpy().astype("float16")
omics_gen_expr_valid = cancer_cell_gen_expr_model(x_valid["Cancer_Cell_Line"].values).numpy().astype("float16")

omics_gen_copy_number_gen_expr_train = np.concatenate([np.expand_dims(omics_copy_number_train, -1),
                                                      np.expand_dims(omics_gen_expr_train, -1)], axis = -1)

omics_gen_copy_number_gen_expr_valid = np.concatenate([np.expand_dims(omics_copy_number_valid, -1),
                                                      np.expand_dims(omics_gen_expr_valid, -1)], axis = -1)

# extract gen methylation features
omics_gen_methyl_train = cancer_cell_gen_methy_model(x_train["Cancer_Cell_Line"].values).numpy().astype("float16")
omics_gen_methyl_valid = cancer_cell_gen_methy_model(x_valid["Cancer_Cell_Line"].values).numpy().astype("float16")

# extract gen mutation features
omics_gen_mut_train = cancer_cell_gen_mut_model(x_train["Cancer_Cell_Line"].values).numpy().astype("float16")
omics_gen_mut_valid = cancer_cell_gen_mut_model(x_valid["Cancer_Cell_Line"].values).numpy().astype("float16")

smile_strings_train = x_train["Smiles"].values.reshape(-1,1)
smile_strings_valid = x_valid["Smiles"].values.reshape(-1,1)

selected_info_common_cell_lines = "../data/CCLE/cellline_list.txt"
selected_info_common_genes = "../data/CCLE/gene_list.txt"

PPI_file = "../data/PPI/PPI_network.txt"

with open(selected_info_common_cell_lines) as f:
    common_cell_lines = [item.strip() for item in f.readlines()]
    
with open("..//data//common_genes.pickle", "rb") as f:
    common_genes = pickle.load(f)
    
idx_dic={}
for index, item in enumerate(common_genes):
    idx_dic[item] = index

ppi_adj_info = [[] for item in common_genes] 
for line in open(PPI_file).readlines():
    gene1,gene2 = line.split('\t')[0],line.split('\t')[1]
    if (gene1 in common_genes) & (gene2 in common_genes):
        if idx_dic[gene1]<=idx_dic[gene2]:
            ppi_adj_info[idx_dic[gene1]].append(idx_dic[gene2])
            ppi_adj_info[idx_dic[gene2]].append(idx_dic[gene1])
            
len(ppi_adj_info)

len(common_genes)

def CelllineGraphAdjNorm(ppi_adj_info,common_genes = common_genes):
    # with open(selected_info_common_genes) as f:
    #     common_genes = [item.strip() for item in f.readlines()]
    nb_nodes = len(common_genes)
    adj_mat = np.zeros((nb_nodes,nb_nodes),dtype='float32')
    # print(adj_mat.shape)
    for i in range(len(ppi_adj_info)):
        # print(i)
        nodes = ppi_adj_info[i]
        for each in nodes:
            adj_mat[i,each] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    norm_adj = NormalizeAdj(adj_mat)
    return norm_adj 

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

import scipy.sparse as sp

ppi_adj = CelllineGraphAdjNorm(ppi_adj_info,common_genes)

ppi_adj = np.expand_dims(ppi_adj,0)


omics_gen_copy_number_gen_expr_train = (ppi_adj@omics_gen_copy_number_gen_expr_train)
omics_gen_copy_number_gen_expr_valid = (ppi_adj@omics_gen_copy_number_gen_expr_valid)

input_smiles_string = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

text_vec = tf.keras.layers.TextVectorization(standardize=None,
    split='character', output_mode = "int")

text_vec.adapt(smile_strings_train, batch_size = 100000)

text_vec_output = text_vec(input_smiles_string)

embedding_layer = tf.keras.layers.Embedding(len(text_vec.get_vocabulary()), 128)
embedding_output = embedding_layer(text_vec_output)
lstm_layer = tf.keras.layers.LSTM(64)
lstm_output_smiles = lstm_layer(embedding_output)

bottleneck_layer_smiles = tf.keras.layers.Dense(32, activation = "relu")

bottleneck_layer_smiles_out = bottleneck_layer_smiles(lstm_output_smiles)

input_rdkit = tf.keras.layers.Input(shape = (drug_features_train.shape[1],))

dense_rdkit_layer = tf.keras.layers.Dense(64, activation = "relu")

rdkit_embs = dense_rdkit_layer(input_rdkit)
dense_rdkit_layer = tf.keras.layers.Dense(32, activation = "relu")
rdkit_embs = dense_rdkit_layer(rdkit_embs)

input_gcn_features = tf.keras.layers.Input(shape = (100, 75))
input_norm_adj_mat = tf.keras.layers.Input(shape = (100, 100))
mult_1 = tf.keras.layers.Dot(1)([input_norm_adj_mat, input_gcn_features])
dense_layer_gcn = tf.keras.layers.Dense(256)
dense_out = dense_layer_gcn(mult_1)
mult_2 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
dense_layer_gcn = tf.keras.layers.Dense(256)
dense_out = dense_layer_gcn(mult_2)
dense_layer_gcn = tf.keras.layers.Dense(100)
mult_3 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
dense_out = dense_layer_gcn(mult_3)
dense_out = tf.keras.layers.GlobalAvgPool1D()(dense_out)
# weights = tf.keras.layers.Softmax()(dense_out)
# weights = tf.keras.layers.Reshape((100, 1))(weights)
# multiply_out = tf.keras.layers.Multiply()([input_gcn_features, weights])
# sum_lambda = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=1))
# gcn_features = sum_lambda(multiply_out)
# dense_gcn = tf.keras.layers.Dense(8)
# gcn_features = dense_gcn(gcn_features)

rdkit_with_smile_embs = tf.keras.layers.Concatenate()([rdkit_embs, bottleneck_layer_smiles_out, dense_out])

drugs_final_emb_layer = tf.keras.layers.Dense(32, activation = "relu")

drugs_smiles_rdkit_embs = drugs_final_emb_layer(rdkit_with_smile_embs)

input_gen_expr = tf.keras.layers.Input(shape = (omics_gen_expr_train.shape[1],2))

l1 = tf.keras.layers.Dense(32)(input_gen_expr)
l2 = tf.keras.layers.Dense(128)(l1)
# mult_11 = tf.keras.layers.Dot(1)([const_input, l2])


dense_layer_gcn1 = tf.keras.layers.Dense(256)
dense_out1 = dense_layer_gcn1(l2)
# mult_21 = tf.keras.layers.Dot(1)([const_input, dense_out1])
dense_layer_gcn1 = tf.keras.layers.Dense(256)
dense_out1 = dense_layer_gcn1(dense_out1)
dense_layer_gcn1 = tf.keras.layers.Dense(256)
# mult_31 = tf.keras.layers.Dot(1)([const_input, dense_out1])
dense_out1 = dense_layer_gcn1(dense_out1)
dense_layer_gcn1 = tf.keras.layers.Dense(256)
# mult_31 = tf.keras.layers.Dot(1)([const_input, dense_out1])
dense_out1 = dense_layer_gcn1(dense_out1)
dense_out1 = tf.keras.layers.GlobalAvgPool1D()(dense_out1)

input_gen_methy = tf.keras.layers.Input(shape = (omics_gen_methyl_train.shape[1],))

gen_methy_layer = tf.keras.layers.Dense(256, activation = "relu")

gen_methy_emb = gen_methy_layer(input_gen_methy)


gen_methy_layer = tf.keras.layers.Dense(128, activation = "relu")
gen_methy_emb = gen_methy_layer(gen_methy_emb)

input_gen_mut = tf.keras.layers.Input(shape = (omics_gen_mut_train.shape[1],))

reshape_gen_mut = tf.keras.layers.Reshape((1, omics_gen_mut_train.shape[1], 1))

reshape_gen_mut = reshape_gen_mut(input_gen_mut)

gen_mut_layer = tf.keras.layers.Conv2D(50, (1, 700), strides=5, activation = "relu")

gen_mut_emb = gen_mut_layer(reshape_gen_mut)

pool_layer = tf.keras.layers.MaxPooling2D((1,5))

pool_out = pool_layer(gen_mut_emb)

gen_mut_layer = tf.keras.layers.Conv2D(30, (1, 5), strides=2, activation = "relu")

gen_mut_emb = gen_mut_layer(pool_out)

pool_layer = tf.keras.layers.MaxPooling2D((1,10))

pool_out = pool_layer(gen_mut_emb)

flatten_layer = tf.keras.layers.Flatten()

flatten_out = flatten_layer(pool_out)


all_omics = tf.keras.layers.Concatenate()([gen_methy_emb, flatten_out, dense_out1])

last_omics_emb = tf.keras.layers.Dense(64, activation = "relu")
final_omics_emb = last_omics_emb(all_omics)

final_drugs_omics = tf.keras.layers.Concatenate()([drugs_smiles_rdkit_embs, final_omics_emb])

final_emb_layer = tf.keras.layers.Dense(32, activation = "relu")


final_emb = final_emb_layer(final_drugs_omics)

final_emb = tf.keras.layers.Dropout(0.05)(final_emb)

final_out_layer = tf.keras.layers.Dense(1)
final_out = final_out_layer(final_emb)

simplecdr = tf.keras.models.Model([input_smiles_string, input_rdkit, input_gcn_features, input_norm_adj_mat, input_gen_expr,
                                   input_gen_methy, input_gen_mut], final_out)



simplecdr.compile(loss = tf.keras.losses.MeanSquaredError(), 
                    optimizer = tf.keras.optimizers.Adam(1e-3), 
                    metrics = [tf.keras.metrics.RootMeanSquaredError()])


history = simplecdr.fit([smile_strings_train, drug_features_train, train_gcn_feats, train_adj_list,
                         omics_gen_copy_number_gen_expr_train, 
                         omics_gen_methyl_train, omics_gen_mut_train], y_train.reshape(-1,1), 
                         
          batch_size = 128, epochs = 10000, verbose = 1,
                         
          validation_data=([smile_strings_valid, drug_features_valid, valid_gcn_feats, valid_adj_list,
                           omics_gen_copy_number_gen_expr_valid, 
                           omics_gen_methyl_valid, omics_gen_mut_valid], y_valid.reshape(-1,1)),
                         

        callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 25, restore_best_weights=True,
                                                       mode = "min"), 
         validation_batch_size = 128, shuffle = True)

simplecdr.save("..//Models//gcn_cdr_both_all_encompass")



val_preds = simplecdr.predict([smile_strings_valid, drug_features_valid, valid_gcn_feats, valid_adj_list,
                           omics_gen_copy_number_gen_expr_valid, 
                           omics_gen_methyl_valid, omics_gen_mut_valid])

import matplotlib.pyplot as plt

plt.scatter(y_valid.reshape(-1,1), val_preds, alpha = 0.1)
plt.axline((0,0), slope = 1, c = "black")
plt.show()

from scipy.stats import pearsonr

res = pearsonr(y_valid.tolist(), val_preds.tolist())

print(res[0][0], flush = True)