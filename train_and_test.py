from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = {"train_batch_size": 4, 
"n_gpu":3, "eval_batch_size": 64, 
'max_answer_length': 50,  
'num_train_epochs': 6, 
'output_dir': "./output/", 'best_model_dir': 
'./output/best_model_bert_bs_8_for_bentham/', 'evaluate_during_training': 
True, 'fp16': False, 'overwrite_output_dir':True,
'use_cached_eval_features':False, 
'save_eval_checkpoints': False, 
'save_model_every_epoch': False, 
'max_seq_length': 384, 
'doc_stride': 128, 'do_lower_case': True, 
'gradient_accumulation_steps': 1, 
'learning_rate': 2e-05, 'multiprocessing_chunksize':10}

model_args_3 = {"train_batch_size": 4, 
"n_gpu":3, "eval_batch_size": 64, 
'max_answer_length': 50,  
'num_train_epochs': 6, 
'output_dir': "./output/", 'best_model_dir': 
'./output/best_model_deberta_bs_4/', 'evaluate_during_training': 
True, 'fp16': False, 'overwrite_output_dir':True,
'use_cached_eval_features':False, 
'save_eval_checkpoints': False, 
'save_model_every_epoch': False, 
'max_seq_length': 384, 
'doc_stride': 128, 'do_lower_case': True, 
'gradient_accumulation_steps': 1, 
'learning_rate': 3e-05, 'optimizer': 'Adam', 'adam_epsilon': 1e-08, 'multiprocessing_chunksize':10}




model = QuestionAnsweringModel('bert', 'bert-large-uncased-whole-word-masking-finetuned-squad', args=model_args)





#with open('./SQuAD_like_HW-SQuAD_train_new_tf_idf_with_transformer_modified.json') as f:
#  train_data = json.load(f)


#print('length of test data------', len(train_data))
#train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)

#dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)



#with open('./BenthamQA_test_new_without_transformer_preprocessed.json') as f:
#  dev_data = json.load(f)

#train 
#model.train_model(train_data, show_running_loss= True, eval_data=dev_data)


with open('./BenthamQA_Squad_like_tf_idf_with_transformer_200_qa.json') as f:
    test_data = json.load(f)


result, predictions = model.eval_model(test_data)
print(result)

#print(predictions)


with open('./BenthamQA_Squad_like_tf_idf_with_transformer_200_qa.txt','w') as f:
    f.write(str(result))
    f.write(str(predictions))


