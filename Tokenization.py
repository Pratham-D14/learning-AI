import tiktoken

# Storing Encoding function instance on the basis of model (model as argument)
enc = tiktoken.encoding_for_model('gpt-4o')

text = 'Hey There I am Pratham! learning how Tokenization works in Generative AI models'

# Giving the prompt for encoding the tokens
tokens = enc.encode(text)
print(tokens)  # printing the encoded tokens

# The Given Encoding Tokens taken for decoding example
textToken = [25216, 3274, 357, 939, 2284, 94125, 0, 7524, 1495, 17951, 2860, 5882, 306, 4140, 1799, 20837, 7015]

decodes = enc.decode(textToken)  # decoding the tokens received
print(decodes)
