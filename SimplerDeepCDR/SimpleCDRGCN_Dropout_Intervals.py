#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pickle


# In[2]:


with open("..//data//valid_items.pickle", "rb") as f:
    valid_items = pickle.load( f)


# In[3]:


from tensorflow.keras import backend as K


# In[4]:


simplecdr = tf.keras.models.load_model("..//Models//combo_cdr_gcn_more_dropout_activated")


# In[5]:


valid_gcn_feats, valid_adj_list,omics_gen_copy_number_gen_expr_valid, omics_gen_methyl_valid, omics_gen_mut_valid = valid_items[0]


# In[6]:


y_valid = valid_items[1]


# In[ ]:


stacked_preds = [simplecdr.predict([ valid_gcn_feats, valid_adj_list,
                           omics_gen_copy_number_gen_expr_valid, 
                           omics_gen_methyl_valid, omics_gen_mut_valid], batch_size = 1000, verbose = 0) for i in range(0, 100)]


# In[ ]:


with open("..//data//stacked_preds_dropout_100.pickle", "wb") as f:
    pickle.dump(stacked_preds, f)

