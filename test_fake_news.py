import unittest
import requests
from flask import json
import time
import pandas as pd
import matplotlib.pyplot as plt

class FakeNewsFlaskTest(unittest.TestCase):

    def setUp(self):
        self.url="http://ece444pra5-env-1.eba-i7c8brd7.us-east-2.elasticbeanstalk.com"

    def test_real_news_prediction(self):
        response = requests.post(self.url+'/predict',
                                 data=json.dumps({'text': 'This is real news'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], 'REAL')

    def test_fake_news_prediction(self):
        response = requests.post(self.url+'/predict',
                                 data=json.dumps({'text': 'This is fake news'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertEqual(data['prediction'], 'FAKE')

    def test_empty_text(self):
        response = requests.post(self.url+'/predict',
                                 data=json.dumps({'text': 'Toronto is a big city'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertIn(data['prediction'], 'REAL')

    def test_invalid_json(self):
        response = requests.post(self.url+'/predict',
                                 data=json.dumps({'text': 'Toronto is in the USA'}),
                                 headers={'Content-Type': 'application/json'})
        data = response.json()
        self.assertIn(data['prediction'], 'FAKE')
    
    def test_100_calls(self):
        response_times = [[],[],[],[]]
        test_cases = [
            {'text': 'This is real news'},
            {'text': 'This is fake news'},
            {'text': 'Toronto is a big city'},
            {'text': 'Toronto is in the USA'}
        ]
        for _ in range(100):
            for i, test in enumerate(test_cases):
              start_time = time.time()
              response = requests.post(self.url+'/predict',
                                  data=json.dumps(test),
                                  headers={'Content-Type': 'application/json'})
              end_time = time.time()
              response_times[i].append(end_time - start_time)

        df = pd.DataFrame(response_times).T
        df.to_csv('response_times.csv', index=False, header=False)
        
        plt.figure()
        df.boxplot()
        plt.show()
        
        self.assertLess(sum(sum(response_times, []))/500, 1.0)

if __name__ == '__main__':
    unittest.main()
