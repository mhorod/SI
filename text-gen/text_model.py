import abc

class TextModel(abc.ABC):
    '''
    Interface of text generating model
    '''

    @abc.abstractmethod
    def generate(self, length):
        '''
        Generate text of given length
        '''
        pass

    @abc.abstractmethod
    def generate_from_prompt(self, prompt, length):
        '''
        Generate text of given length starting with prompt
        '''
        pass

    @abc.abstractmethod
    def perplexity(self, text):
        '''
        Calculate perplexity of given text - how much this model is surprised by the text
        '''
        pass