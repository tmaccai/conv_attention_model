Model has been saved, you can skip training step and run the test step directly.

1. Train
CNN: python CNN.py train
CRAN: python CRAN.py train

2. Test
CNN: python CNN.py test test_file_name
CRAN: python CRAN.py test test_file_name

Input: a pos.txt like txt file, contains comments for test
Output: a txt file containing three columns [sentence, predicted label, key feature]
