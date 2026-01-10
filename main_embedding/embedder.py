import torch
from blingfire import text_to_sentences_and_offsets
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceExtractor:
    def _extract_sentences(self, context):
        """
        Extracts and returns sentences from given context.
        Parameters:
            context (str): Input document context.
        Returns:
        """
        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(context)

        # Initialize a list to store sentences
        sentences = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = context[start:end]
            sentences.append(sentence)

        return sentences

class SentenceEmbedder:
    def __init__(self, model_name, accelerator=None):
        self.sentence_model = SentenceTransformer(
            model_name,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        if accelerator:
            self.sentence_model.to(accelerator.device)
        self.sentence_extractor = SentenceExtractor()

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=128,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def estimate_importance(self, context, query):
        sentences = self.sentence_extractor._extract_sentences(context)
        sentences_embeddings = self.calculate_embeddings(sentences)
        query_embeddings = self.calculate_embeddings(query)
        # Calculate cosine similarity between query and chunk embeddings,
        cosine_scores = (sentences_embeddings * query_embeddings).sum(1)
        
        return sentences, (-cosine_scores).argsort()


def extract_sentences_from_encoder_input_ids(
    encoder_input_ids: List[List[int]],
    tokenizer,
    sentence_extractor,
):
    context_list = []
    sentences_list = []
    deviation_list = []
    for _encoder_input_ids in encoder_input_ids:
        encoder_length = len(_encoder_input_ids)
        context = tokenizer.decode(_encoder_input_ids)

        length_deviation = len(tokenizer.encode(context)) - encoder_length
        sentences = sentence_extractor._extract_sentences(context)

        context_list.append(context)
        sentences_list.append(sentences)
        deviation_list.append(length_deviation)
    
    return context_list, sentences_list, deviation_list


def get_sentence_begin_indices(
    encoder_input_ids: List[List[int]],
    tokenizer,
    sentence_extractor,
):
    sentence_begin_indices_list = []
    context_list, sentences_list, deviation_list = extract_sentences_from_encoder_input_ids(encoder_input_ids, tokenizer, sentence_extractor)
    number_of_sentences = []
    idx1 = 0
    sentence_number = 0
    
    # * extract each sentences' begin index in encoder_input_ids, considering length deviation
    for context, _sentences_list, deviation in zip(context_list, sentences_list, deviation_list):
        sentence_begin_indices = [{'idx1': idx1, 'idx2': 0}]
        former_index1 = 0
        for i in range(len(_sentences_list) - 1):

            sentence1 = _sentences_list[i]
            sentence2 = _sentences_list[i + 1]

            split1 = context[former_index1:].split(sentence1)[0] + sentence1
            split1_total = context[:former_index1] + split1  
            former_index1 += len(split1)

            split2 = context[former_index1:].split(sentence2)[0] + sentence2
            split2_total = context[:former_index1] + split2

            indices1 = tokenizer.encode(split1_total)
            indices2 = tokenizer.encode(split2_total)
            if split1_total != context[:former_index1]:
                print('former_index1', former_index1)
                exit()
            for j in range(len(indices1)):
                if j - deviation > 4095: #FIXME assume lm_max_length=4096
                    print('deviation', deviation)
                    exit()
                if indices1[j] != indices2[j]:
                    sentence_begin_indices.append({'idx1': idx1, 'idx2': j - deviation})
                    break
            if (j == len(indices1) - 1) and (indices1[j] == indices2[j]):
                sentence_begin_indices.append({'idx1': idx1, 'idx2': j + 1 - deviation})
        
        # append the total length at the end for subsequent calculations
        sentence_begin_indices.append({'idx1': idx1, 'idx2': len(encoder_input_ids[idx1])})
        sentence_number += len(_sentences_list)

        number_of_sentences.append(sentence_number)

        idx1 += 1

        sentence_begin_indices_list.extend(sentence_begin_indices)
    
    return  number_of_sentences, sentences_list, sentence_begin_indices_list


def get_sentence_priority_list(
    sentences_list: List[List[str]],
    sentence_embedder,
    prompt,
    number_of_sentences,
    dataset_name,
):
    # * Convert List[List[str]] to List[str]
    sentences = []
    for _sentences_list in sentences_list: 
        sentences.extend(_sentences_list)

    # * Extract query, maybe not fitted at all time
    query = prompt.split('Question: ')[1].split("\nAnswer:")[0]

    sentences_embeddings = sentence_embedder.calculate_embeddings(sentences)
    query_embeddings = sentence_embedder.calculate_embeddings(query)
    # * Calculate cosine similarity between query and chunk embeddings,
    cosine_scores = (sentences_embeddings * query_embeddings).sum(1)

    priority_list = (-cosine_scores).argsort()

    # * Handle the impact of changes in idx caused by inserting total length before processing
    for number in number_of_sentences[::-1]:
        temp = []
        for sentence_id in priority_list:
            if sentence_id >= number:
                temp.append(sentence_id + 1)
            else:
                temp.append(sentence_id)
        priority_list = temp

    # * Get sentences_ids_list in encoder_input_ids split way for encoder_indices generation
    sentences_ids = sorted(priority_list)
    temp = [] 
    sentences_ids_list = []
    for idx in range(len(sentences_ids)):
        temp.append(sentences_ids[idx])
        if (idx == len(sentences_ids) - 1) or (sentences_ids[idx] + 1 != sentences_ids[idx + 1]):
            sentences_ids_list.append(temp)
            temp = []
    
    return priority_list, sentences_ids_list


# merge two list
def merge(list1, list2):
    combined_set = set(list1) | set(list2)
    return list(combined_set)
