from skills.skill_base import SkillBase
from sklearn import linear_model
import os
import pandas as pd
import random 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

data_path = os.path.join(os.environ.get('DATA_STORAGE_PATH'), 'soggy_egg_timer')

class SoggyEggTimerSkill(SkillBase):
    def __init__(self) -> None:
        super().__init__()
        print(f"Created {self.__class__.__name__} skill")

        random.seed(int(datetime.now().timestamp()))

        self.trained_model = None
        self.last_weight = None
        self.last_predicted_time = None
        self.last_predicted_time_pretty = None
        self.last_msg = None 
        self.__train_model()
        # self.__plot()
        
        # print("Time for an egg of 61 gr:", self.__calc_datapoint(61), "seconds")


        print(f"Initialized {self.__class__.__name__} skill")


    def __calc_datapoint(self, x):
        x = self.weight_scaler.transform([[x]])[0][0]
        y = self.trained_model.predict([[x, 1]])[0]
        y = self.cooktime_scaler.inverse_transform([[y]])[0][0]
        return y

    def msg_callback(self, msg):
        """ This function will be called for every message that matches the intent specified in the config.yaml of this skill """
        print(f"{self.__class__.__name__} skill received message:", msg)

        self.last_msg = msg 

        # query the cooking time for a soggy egg
        if msg.intent == "time_soggy_egg":
            if "weight" not in msg.matches:
                self.output_message(
                    text="I'm sorry, I didn't understand how much your egg weighs. Could you try again?", reply_to=self.last_msg)
                return
            else:
                weight = msg.matches["weight"]
                try:
                    weight = int(weight)
                except:
                    self.output_message(
                        text="I'm sorry, I didn't understand how much your egg weighs. Could you try again?", reply_to=self.last_msg)
                    return
                self.__calc_egg_time(weight)

        # rate the soggy egg 
        elif msg.intent == "rate_soggy_egg":
            if self.last_weight is None or self.last_predicted_time is None:
                self.output_message(
                    text="I'm sorry, I don't know what egg you are talking about! Please ask me first on how long to cook your egg.", reply_to=self.last_msg)
                return
            if "rating" not in msg.matches:
                self.output_message(
                    text="I'm sorry, I didn't understand how you liked your egg. Could you give it a number between 0 and 10?", reply_to=self.last_msg)
                return
            else:
                rating = msg.matches["rating"]
                try:
                    rating = int(rating)
                    assert rating < 11 and rating > -1
                except:
                    self.output_message(
                        text="I'm sorry, I didn't understand how you liked your egg. Could you give it a number between 0 and 10?", reply_to=self.last_msg)
                    return
                self.__add_data(self.last_weight, self.last_predicted_time, rating)
                self.last_weight = None 
                self.last_predicted_time = None 
    
    def __load_datafile(self):
        baseline_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "egg_data.csv")

        # preferably we use the file stored in the volume, but if it doesn't exist, we copy the baseline data file to the volume
        self.data_file = os.path.join(data_path, "egg_data.csv")
        
        # check if self.data_file exists
        if not os.path.exists(data_path):
            print(f"{self.__class__.__name__} data file doesn't exist, copying baseline data to it")
            # create a file called egg_data.csv in path and copy the baseline data to it
            os.makedirs(data_path)
            self.egg_data = pd.read_csv(baseline_data_file, sep=",")
            self.egg_data.to_csv(self.data_file, index=False, sep=",")
        else:
            print("Found pre-existing egg_data.csv data file")

        self.egg_data = pd.read_csv(self.data_file, sep=",")


    def __normalize_data(self):
        """ Normalize the data in the egg_data.csv file """
                
        # normalize rating from 0-10 to 0-1
        self.egg_data['rating_normalized'] = self.egg_data['rating'] / 10

        # normalize the weight feature
        self.weight_scaler = MinMaxScaler()
        wgr = [[x] for x in self.egg_data['weight_gr'].to_list()]
        self.weight_scaler.fit(wgr)
        self.egg_data['weight_gr_normalized'] = self.weight_scaler.transform(wgr)

        # normalize the cook seconds feature
        self.cooktime_scaler = MinMaxScaler()
        cs = [[x] for x in self.egg_data['cook_seconds'].to_list()]
        self.cooktime_scaler.fit(cs)
        self.egg_data['cook_seconds_normalized'] = self.cooktime_scaler.transform(cs)
        
    

    def __train_model(self):

        self.__load_datafile()
        self.__normalize_data()

        # split the eggs dataframe into a training and test set
        train = self.egg_data.sample(frac=0.8, random_state=200)
        test = self.egg_data.drop(train.index)

        print(test.head())

        # fit a linear function to it
        self.trained_model = linear_model.LinearRegression()

        # fit regr on two features: weight and rating, and predict the cooking time
        self.trained_model.fit(train[['weight_gr_normalized', 'rating_normalized']], train['cook_seconds_normalized'])

        real = test['cook_seconds_normalized'].to_list()
        predicted = self.trained_model.predict(test[['weight_gr_normalized', 'rating_normalized']]).tolist()
        mse = mean_squared_error(real, predicted)

        print(f"Trained soggy egg model. Mean Squared Error of {len(real)} test datapoints: {mse}")
        # print("Model coefficients:", self.trained_model.coef_)


    def __calc_egg_time(self, weight):
        if self.trained_model is None:
            self.__train_model()

        # normalize the weight
        weight = [[weight]]
        weight_normalized = self.weight_scaler.transform(weight)[0][0]

        # predict the cooking time
        time_normalized = self.trained_model.predict([[weight_normalized, 1]])[0]
        time = self.cooktime_scaler.inverse_transform([[time_normalized]])[0][0]
        self.last_predicted_time = time

        # convert seconds to entire minutes and the rest as seconds
        minutes = int(time // 60)
        seconds = int(time % 60)

        msg = random.choice([
            "Your egg should be perfect if you cook it for ",
            "Try cooking it for ",
            "I think you should cook it for ",
            "Perfect soggyness will be achieved if you cook it for ",
            "You know what, this egg deserves "])
        if minutes > 0:
            msg += f"{minutes} minutes and {seconds} seconds."
            self.last_predicted_time_pretty = f"{minutes} minutes and {seconds} seconds."
        else:
            msg += f"{seconds} seconds."
            self.last_predicted_time_pretty = f"{seconds} seconds."

        self.last_weight = weight 

        self.output_message(text=msg, reply_to=self.last_msg)


    def __add_data(self, weight, time, rating):
        try:
            rating = int(rating)
        except:
            self.output_message(
                text="I'm sorry, I didn't understand your rating. Could you give it a number between 0 and 10?", reply_to=self.last_msg)
            return

        # add the data to the pandas dataframe
        datapoint = pd.Series({'cook_seconds': time,
                        'weight_gr': weight,
                        'rating': rating})
        self.egg_data = pd.concat([self.egg_data, datapoint.to_frame().T])

        self.egg_data = self.egg_data.drop_duplicates()

        # save the data to the csv file
        self.egg_data.to_csv(self.data_file, index=False, sep=",")
        print("Wrote data to egg data file")

        # retrain the model
        self.__train_model()

        # output a message
        msg = None 
        if rating > 7: 
            msg = random.choice([
                "Yay! I'm so happy you liked it!", 
                "You can't imagine how happy I am that you liked it! God bless having a simple purpose in life.",
                "I'm so happy you liked it!",
                "I'm so happy you liked it! I'm going to tell my creator about this.",
                "Ha! You thought I was just a dumb egg timer, but I'm actually a machine learning algorithm!",
                "I may not be able to exit this stupid box, but atleast I can make a perfect egg!",
                f"The answer to the ultimate question of life, the universe, and everything is 42. But the answer to the question of how to cook a perfect egg that weighs {self.last_weight} grams is {self.last_predicted_time_pretty} Thank you for the input."])
        else:
            msg = random.choice([
                "Thank you for the input. I updated the model for extra future yum.",
                "I will remember this for next time. :D",
                "As they say, feedback is the breakfast of champions. Thank you!",
                "Let's make this world a better place, one soggy egg at a time."
                "I'm sorry you didn't like it. I will remember this for next time.",
                "Oh no! I'm sorry you didn't like it. Hopefully I can make it better next time."])
        self.output_message(text=msg, reply_to=self.last_msg)

    def __plot(self):
        import matplotlib.pyplot as plt

        # prepare axes for the plot 
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Cooking duration (seconds)')
        ax1.set_ylabel('Egg weight (gram)')
        ax1.set_title('The perfect soggy egg')

        c = self.egg_data.rating
        plt.scatter( self.egg_data['cook_seconds'], self.egg_data['weight_gr'] , c=c, cmap = 'RdYlGn', s=70) 
        # plot axes etc
        ax1.set_ylim((40, 75))
        ax1.set_xlim((300, 480))
        cbar = plt.colorbar()
        cbar.set_label('How great was the egg?')

        # plot the learned linear function
        test_y = [[item, 1] for item in list(range(40, 75, 5))]

        # normalize for the linear regression model
        test_y_normalized = []
        for item in test_y:
            print(item)
            item_scaled = self.weight_scaler.transform([[item[0]]])[0][0]
            test_y_normalized.append([item_scaled, item[1]])

        # calc predicated value  of test set
        predicted_normalized = self.trained_model.predict(test_y_normalized)
        # denormalize so we can plot it
        predicted = self.cooktime_scaler.inverse_transform([[item] for item in predicted_normalized])
        predicted = [item[0] for item in predicted]

        plt.plot(predicted, [item[0] for item in test_y], color='blue', linewidth=3)

        plt.show()

        

def create_skill():
    return SoggyEggTimerSkill()
