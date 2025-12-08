import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TFAutoModel
from transformers import TFBertForMaskedLM, BertTokenizer
import pandas as pd
import keras

# ------------- Code based on code from Hollenstein & Beinborn (2021) -------------
# source: https://github.com/beinborn/relative_importance/blob/main/extract_model_importance/extract_saliency.py

def ensure_built(model, tokenizer, min_len: int = 2):

    """Run a tiny forward pass so all layers/variables are created."""

    # Create a minimal dummy input with [CLS] ... [SEP]
    ids = tf.constant([[tokenizer.cls_token_id, tokenizer.sep_token_id]], dtype=tf.int32)
    attn = tf.ones_like(ids)
    # Call the model once; this will also build the embeddings module
    _ = model(input_ids=ids, attention_mask=attn, training=False)

def get_word_embedding_table_built(emb_module):

    """
    Return the tf.Variable for the *word* embedding table (shape [vocab_size, hidden_size])
    from a TFBertEmbeddings module, robust across TF/Transformers versions.
    This assumes the model has been built (variables created).
    """

    # 1) Try to find by variable name first (most reliable)
    # Typical names include ".../word_embeddings/embeddings:0" or ".../token_embeddings/embeddings:0"
    for v in emb_module.weights:
        name = v.name.lower()
        if ("word_embeddings" in name or "token_embeddings" in name) and len(v.shape) == 2:
            return v  # shape [vocab_size, hidden_size]

    # 2) Fallback: pick the 2D variable with the largest first dimension (vocab_size)
    two_d_vars = [v for v in emb_module.weights if len(v.shape) == 2]
    if not two_d_vars:
        raise RuntimeError("No 2D variables found in embeddings; is the model built?")
    word_table = max(two_d_vars, key=lambda v: int(v.shape[0]))
    return word_table

def compute_sensitivity_bert(model: TFBertForMaskedLM,
                            embedding_matrix,
                            tokenizer: BertTokenizer,
                            text: list[str]):

    embedding_matrix = get_word_embedding_table_built(embedding_matrix)
    vocab_size = embedding_matrix.shape[0]
    # vocab_size = int(embedding_matrix.weights[0].shape[0])
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    sensitivity_data = []

    # iteractively mask each token in the input
    for masked_token_id in range(len(token_ids)):

        if masked_token_id == 0:
            sensitivity_data.append({'token': '[CLS]', 'sensitivity': [1] + [0] * (len(token_ids) - 1)})

        elif masked_token_id == len(token_ids) - 1:
            sensitivity_data.append({'token': '[SEP]', 'sensitivity': [0] * (len(token_ids) - 1) + [1]})

        # original token at this position
        else:
            target_token = tokenizer.convert_ids_to_tokens(token_ids[masked_token_id])
            # create a masked version of the ids
            masked_ids = token_ids.copy()
            masked_ids[masked_token_id] = tokenizer.mask_token_id
            # (1, seq)
            token_ids_tensor = tf.constant([masked_ids], dtype=tf.int32)
            # one hot (1, seq, vocab)
            token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)

            # Build a mask to select the logit of the original token at the masked position
            output_mask = np.zeros((1, len(token_ids), vocab_size))
            output_mask[0, masked_token_id, token_ids[masked_token_id]] = 1
            output_mask_tensor = tf.constant(output_mask, dtype='float32')

            # Compute gradient of the logits of the correct target, w.r.t. the input
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(token_ids_tensor_one_hot)
                inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix)
                predict = model({"inputs_embeds": inputs_embeds}).logits
                predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)

            # compute the sensitivity and take l2 norm
            sensitivity_non_normalized = tf.norm(tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot),
                                                 axis=2)

            # Normalize by the max
            sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_max(sensitivity_non_normalized))
            sensitivity = sensitivity_tensor[0].numpy().tolist()

            sensitivity_data.append({'token': target_token, 'sensitivity': sensitivity})

    return sensitivity_data

def compute_sensitivity_gpt(model: TFGPT2LMHeadModel,
                            embedding_matrix:keras.src.layers.core.embedding.Embedding,
                            tokenizer: GPT2Tokenizer,
                            text: list[str]) -> list[dict]:

    """
    Compute sensitivity score of each previous word in relation to each upcoming word.
    :param model: language model
    :param embedding_matrix: embedding layer of langauge model
    :param tokenizer: tokenizer of language model
    :param text: text string
    :return: list with saliency data (token, token_id, and distributed saliency - each entry is the saliency score of the previous token in relation to the current token)
    """

    vocab_size = embedding_matrix.input_dim
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    # print(text)
    # print(len(token_ids))
    sensitivity_data = []

    for token_id in range(len(token_ids)):

        target_token = tokenizer.convert_ids_to_tokens(token_ids[token_id])
        # print(token_id, target_token)

        # tensor for model prediction
        # gpt is not bidirectional, thus tensor up until the current token
        token_ids_tensor = tf.constant([token_ids[:token_id+1]], dtype='int32')
        token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)

        # tensor with correct output
        output_mask = np.zeros((1, len(token_ids[:token_id+1]), vocab_size))
        output_mask[0, token_id, token_ids[token_id]] = 1
        output_mask_tensor = tf.constant(output_mask, dtype='float32')

        # compute gradient of the logits of the correct target, w.r.t. the input
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(token_ids_tensor_one_hot)
            inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix.weights)
            predict = model({"inputs_embeds": inputs_embeds}).logits
            predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)

        # compute the sensitivity and take l2 norm
        sensitivity_non_normalized = tf.norm(tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot), axis=2)

        # normalize by the max
        sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_max(sensitivity_non_normalized))
        sensitivity = sensitivity_tensor[0].numpy().tolist()
        # print(sensitivity, '\n')
        sensitivity_data.append({'token': target_token, 'sensitivity': sensitivity})

    return sensitivity_data

def extract_relative_saliency(model: TFGPT2LMHeadModel | TFBertForMaskedLM,
                              embeddings,
                              tokenizer:GPT2Tokenizer | BertTokenizer,
                              text,
                              model_name:str):

    """
    Compute saliency values for each word in the text.
    :param model: language model
    :param embeddings: embedding layer of language model
    :param tokenizer: tokenizer of language model
    :param text: text string
    :return: the resulting tokens, summed saliency, and distributed saliency values (each previous token relative to current token)
    """

    if 'gpt' in model_name:

        sensitivity_data = compute_sensitivity_gpt(model, embeddings, tokenizer, text)
        distributed_sensitivity = [entry["sensitivity"] for entry in sensitivity_data]
        tokens = [entry["token"] for entry in sensitivity_data]
        # For each token, sum the sensitivity values it has with all other tokens
        distributed_sensitivity_updated = []
        for item, dist_s in enumerate(distributed_sensitivity):
            dist = [s for s in dist_s]
            for i in range(len(dist), len(tokens)):
                dist.append(0) # make all arrays same length (length of token sequence)
            distributed_sensitivity_updated.append(dist)
        saliency_sum = np.sum(distributed_sensitivity_updated, axis=0)

    elif 'bert' in model_name:

        sensitivity_data = compute_sensitivity_bert(model, embeddings, tokenizer, text)
        distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
        tokens = [entry["token"] for entry in sensitivity_data]
        saliency_sum = np.sum(distributed_sensitivity, axis=0)

    else:
        raise ValueError('Unsupported model')

    return tokens, saliency_sum, distributed_sensitivity

def extract_all_saliency(model: TFGPT2LMHeadModel | TFBertForMaskedLM,
                         embeddings,
                         tokenizer: GPT2Tokenizer | BertTokenizer,
                         texts:list[str],
                         saliency_path,
                         model_name: str)->pd.DataFrame:

    """
    Compute saliency values for each word in each text.
    :param model: language model
    :param embeddings: embedding layer of language model
    :param tokenizer: tokenizer of language model
    :param texts: texts to compute saliency
    :return: dataframe with saliency values
    """

    all_text_ids, all_tokens, all_saliency_sum, all_dist_saliency = [], [], [], []

    for i, text in enumerate(texts):
        # for each text, compute gradient saliency of each previous token relative to each token
        tokens, saliency_sum, dist_saliency = extract_relative_saliency(model, embeddings, tokenizer, text, model_name)
        # remove CLS and SEP tokens
        saliency_sum = saliency_sum[1:len(tokens) - 1]
        dist_saliency = dist_saliency[1:len(tokens) - 1]
        tokens = tokens[1:len(tokens)-1]
        all_text_ids.extend([i for token in tokens])
        all_tokens.extend(tokens)
        all_saliency_sum.extend(saliency_sum)
        all_dist_saliency.extend(dist_saliency)

        df = pd.DataFrame({'trial_id': all_text_ids,
                           'token': all_tokens,
                           'distributed_saliency': all_dist_saliency,
                           'saliency_sum': all_saliency_sum})

        # store temporary result
        df.to_csv(saliency_path, index=False)

    return df

def merge_multi_tokens(words, word_ids, pos_tags, summed_saliencies, tokenizer, model_name):

    adjusted_saliencies = np.zeros(len(words))
    current_token_position = 0

    for i, word in enumerate(words):
        # if not punctuation, nor starting the text, add whitespace. Whether the word starts with a whitespace makes a difference as to whether it will be a multi-token or not
        if 'gpt2' in model_name and pos_tags[i] != 'PUNCT' and word_ids[i] > 0:
                word = ' ' + word
        tokenized_word = tokenizer.encode(word, add_special_tokens=False)
        # print(word, tokenizer.convert_ids_to_tokens(tokenized_word), current_token_position, len(tokenized_word))
        adjusted_saliency = np.sum(summed_saliencies[current_token_position:current_token_position+len(tokenized_word)])
        # print(adjusted_saliency, i)
        adjusted_saliencies[i] = adjusted_saliency
        current_token_position += len(tokenized_word)
        # make sure saliency of punctuation get added to previous word (" quotation mark no way to know if to previous or upcoming word, so not included)
        if pos_tags[i] == 'PUNCT':
            if word in [',','.','?','!', ';', ':']:
                adjusted_saliencies[i-1] += adjusted_saliency

    return adjusted_saliencies

def normalize_saliency(group):

    norm_saliency = (np.array(group['saliency'].tolist()) - np.min(group['saliency'].tolist())) / (np.max(group['saliency'].tolist()) - np.min(group['saliency'].tolist()))
    group['norm_saliency'] = norm_saliency
    return group

def reassign_word_id(group):

    word_ids = [i for i, word in enumerate(group['word'].tolist())]
    group['word_id'] = word_ids
    return group

def calculate_saliency_values(texts_df:pd.DataFrame, words_df:pd.DataFrame, model_name:str, saliency_path) -> pd.DataFrame:

    """
    Compute gradient saliency values for a given dataset.
    :param texts_df: dataframe with texts
    :param words_df: dataset with text words for which we want to compute saliency.
    :param model_name: name of language model with which to compute saliency.
    :return: dataframe with lm tokens, distributed sensitivity and summed saliencies; and dataframe with saliency values aligned to words in text.
    """

    texts = texts_df.text.values
    words = words_df.word.values
    word_ids = words_df.word_id.values
    pos_tags = words_df.pos_tag.values

    if 'gpt2' in model_name:

        model = TFGPT2LMHeadModel.from_pretrained(model_name, from_pt=True, output_attentions=True, return_dict_in_generate=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        embeddings = model.get_input_embeddings()

    elif 'bert' in model_name:

        model = TFBertForMaskedLM.from_pretrained(model_name, from_pt=True, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        ensure_built(model,tokenizer)
        embeddings = model.get_input_embeddings()

    else:
        raise ValueError(f'Model {model_name} not supported.')

    print(f'Extract saliency with {model_name}')
    saliency_df = extract_all_saliency(model, embeddings, tokenizer, texts, saliency_path, model_name)
    # saliency_df = pd.read_csv(saliency_path)
    saliencies = saliency_df.saliency_sum.values
    adjusted_saliencies = merge_multi_tokens(words, word_ids, pos_tags, saliencies, tokenizer, model_name)
    words_df['saliency'] = adjusted_saliencies
    # remove punctuation from list
    words_df = words_df[words_df['pos_tag'] != 'PUNCT']
    # re-assign word_ids
    words_df = words_df.groupby(['trial_id']).apply(lambda group: reassign_word_id(group)).reset_index(drop=True)
    # normalize saliencies
    words_df = words_df.groupby(['trial_id']).apply(lambda group: normalize_saliency(group)).reset_index(drop=True)
    # # create bins for saliency values
    # # bin 1 (0 - .008), bin 1 (.008 - .18), and bin 2 (.018 - 1.)
    # words_df['norm_saliency_bin'] = pd.qcut(words_df['norm_saliency'], q=3, labels=False)

    return saliency_df, words_df

