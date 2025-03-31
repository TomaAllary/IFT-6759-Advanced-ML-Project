import random

import torch


FIXED_SHAPE = (1400, 768) #(padded_size, nb_features)


def get_example_sets():
    example_sets_audio = []
    example_sets_text = []
    for i in range(30):
        audio_lenght = random.randint(10,35)
        text_lenght = random.randint(10, 35)

        audio_code = torch.randn(audio_lenght, FIXED_SHAPE[1]) # n X nb_features
        text_code = torch.randn(text_lenght, FIXED_SHAPE[1]) # m X nb_features

        example_sets_audio.append(audio_code)
        example_sets_text.append(text_code)

    return example_sets_audio, example_sets_text

def add_padding_to_code(code_list, expected_first_dim_size):
    """
    Change list of uneven merged_codes of shape ((n_i + m_i), nb_features)
    to a single tensor of shape (B, expected_first_dim_size, nb_features) applying
    padding or truncating for even dimensions.
    """

    # make sure max_length is at least expected_first_dim_size
    code_list.append(torch.zeros(expected_first_dim_size, FIXED_SHAPE[1]))

    print(f"shape of empty code: {code_list[-1].shape}")

    # Apply padding. shape will be (B, longest_code, nb_features)
    data = torch.nn.utils.rnn.pad_sequence(code_list, batch_first=True, padding_value=0.0, padding_side='right')
    # Apply truncating using slicing (and remove our previous zero tensor)
    data = data[:-1, 0:expected_first_dim_size, :]

    return data

def format_data(conversation_audio_codes, conversation_text_codes):

    print(f"conversation_audio_codes: {conversation_audio_codes}")
    print(f"conversation_text_codes: {conversation_text_codes}")
    for i in range(4):
        print(f"audio_code_{i}: {conversation_audio_codes[i].shape}")
        print(f"text_code_{i}: {conversation_text_codes[i].shape}")

    # MERGE
    merged_codes_list = []
    for i in range(len(conversation_audio_codes)):
        audio_code = conversation_audio_codes[i]
        text_code = conversation_text_codes[i]

        # n_i X nb_features & m_i X nb_features -> (n_i + m_i) , nb_features
        merged = torch.cat((audio_code, text_code), dim=0)
        merged_codes_list.append(merged)

    for i in range(4):
        print(f"merged_code_{i}: {merged_codes_list[i].shape}")

    # PAD / TRUNC
    data = add_padding_to_code(merged_codes_list, FIXED_SHAPE[0])

    print(f"data shape: {data.shape}")

    return data # tensor of shape (B, FIXED_SHAPE[0], nb_features )


#### TEST #####
example_sets_audio, example_sets_text = get_example_sets()
data = format_data(example_sets_audio, example_sets_text)