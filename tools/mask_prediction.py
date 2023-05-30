import torch
try:
    from chech_conjug import check_conjugation
except:
    from tools.chech_conjug import check_conjugation


def mask_prediction(sentence_available, tokenizer, model, device, conj_act_token_id):
    encoded_sentence = tokenizer.encode(sentence_available, return_tensors="pt").to(device)
    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]
    token_logits = model(encoded_sentence)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    indice_act, sorted, probas_tok = get_indice_act(mask_token_logits, conj_act_token_id)
    top_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
    predicted_token=tokenizer.decode([top_token[0]])
    return predicted_token, mask_token_logits[0], indice_act, probas_tok




def get_indice_act(token_logits, conj_act_token_id):

    probas = torch.softmax(token_logits[0], dim=0)

    sorted, indices = torch.sort(probas, descending=True)

    indx = (indices == conj_act_token_id).nonzero()

    return indx.item(), indices, sorted[indx]





def batch_mask_prediction(batch_sentences, tokenizer, model, device, size_batches = 8):

    all_predictions = []

    nb_batches = len(batch_sentences) // size_batches
    for k in range(nb_batches):
        current_batch = batch_sentences[k*size_batches:(k+1)*size_batches]
        predicted_tokens = encode_batch(current_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)



    if len(batch_sentences) % size_batches != 0:
        last_batch = batch_sentences[nb_batches*size_batches:]
        predicted_tokens = encode_batch(last_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)


    return all_predictions

### vedere bene:
def make_and_encode_batch(current_batch, tokenizer, model, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found):
    current_found = found
    good_pred = 0
    detail_verbs = []

    predictions = encode_batch(current_batch, tokenizer, model, device)
    new_sentence = None

    for i, prediction_available in enumerate(predictions):
        good_verb = batch_verbs[i]




        if check_conjugation(good_verb, prediction_available):
            #print(f"goooood: ({good_verb}) - predict :  ({prediction_available}) : TRUE")
            detail_verbs.append(good_verb)
            good_pred += 1
            good_dico = {"name_available": name_available, "profession_available": profession_available,
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj}

            if not current_found:
                new_sentence = good_dico

                current_found = True
                #if not complete_check: ########
                #    break

        else:
            ppp = 0
            #print(f"goooood: ({good_verb}) - predict :  ({prediction_available}) : False")
    return new_sentence, current_found, good_pred, detail_verbs

### Guardare bene questo!!!!
def encode_batch(current_batch, tokenizer, model, device):

    with torch.no_grad():
        encoded_sentence = tokenizer.batch_encode_plus(current_batch,padding=True,  return_tensors="pt").to(device) # encode frasi
        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id) # recupero l'indice del mask nella frase x sapere quale token devo vedere
        #print(mask_tokens_index)
        tokens_logits = model(**encoded_sentence) # recup vettori
        #print(tokens_logits)
        #print(tokens_logits['logits'].shape)

        mask_tokens_logits = tokens_logits['logits'][ mask_tokens_index] # logits = token (prob x ogni parola del vocab), ne prendi solo quello con indice del mask
        #print(mask_tokens_logits.shape)
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist() ### recup i k logit + alti
        #print(top_tokens)
        predicted_tokens = tokenizer.batch_decode(top_tokens) # decode il batch di token (indicati come indici in riferim al vocab)
        #print(predicted_tokens)

    return predicted_tokens