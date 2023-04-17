from skills.skill_base import SkillBase
from sklearn import linear_model
import os
import pandas as pd
import random 

data_path = os.environ.get('DATA_STORAGE_PATH')

class SoggyEggTimerSkill(SkillBase):
    def __init__(self) -> None:
        super().__init__()
        print(f"Created {self.__class__.__name__} skill")

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

        self.egg_data = pd.read_csv(self.data_file, sep=",")

        self.trained_model = None
        self.last_weight = None
        self.last_predicted_time = None
        self.last_predicted_time_pretty = None
        self.__train_model()

        print(f"Initialized {self.__class__.__name__} skill")

    def msg_callback(self, msg):
        """ This function will be called for every message that matches the intent specified in the config.yaml of this skill """
        print(f"{self.__class__.__name__} skill received message:", msg)

        # query the cooking time for a soggy egg
        if msg.intent == "time_soggy_egg":
            if "weight" not in msg.matches:
                self.output_message(
                    "I'm sorry, I didn't understand how much your egg weighs. Could you try again?")
                return
            else:
                weight = msg.matches["weight"]
                try:
                    weight = int(weight)
                except:
                    self.output_message(
                        "I'm sorry, I didn't understand how much your egg weighs. Could you try again?")
                    return
                self.__calc_egg_time(weight)

        # rate the soggy egg 
        elif msg.intent == "rate_soggy_egg":
            if self.last_weight is None or self.last_predicted_time is None:
                self.output_message(
                    "I'm sorry, I don't know how long your egg was cooked. Could you try again?")
                return
            if "rating" not in msg.matches:
                self.output_message(
                    "I'm sorry, I didn't understand how you liked your egg. Could you give it a number between 0 and 10?")
                return
            else:
                rating = msg.matches["rating"]
                try:
                    rating = int(rating)
                    assert rating < 11 and rating > -1
                except:
                    self.output_message(
                        "I'm sorry, I didn't understand how you liked your egg. Could you give it a number between 0 and 10?")
                    return
                self.__add_data(self.last_weight, self.last_predicted_time, rating)


    def __train_model(self):
        # get the perfect stuff
        best_eggs = self.egg_data[self.egg_data['rating'] > 8]
        best_eggs = best_eggs.reset_index()

        # make a trainingset of the best eggs
        y = []
        x = []
        weights = best_eggs['rating'].to_list()
        for i in range(best_eggs.shape[0]):
            y.append([best_eggs.loc[i, :]['weight_gr']])
            x.append(best_eggs.loc[i, :]['cook_seconds'])

        # fit a linear function to it
        self.trained_model = linear_model.LinearRegression()
        self.trained_model.fit(y, x, weights)

        print("Trained soggy egg model")


    def __calc_egg_time(self, weight):
        if self.trained_model is None:
            self.__train_model()

        time = int(self.trained_model.predict([[weight]])[0])
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

        self.output_message(msg)


    def __add_data(self, weight, time, rating):
        try:
            rating = int(rating)
        except:
            self.output_message(
                "I'm sorry, I didn't understand your rating. Could you give it a number between 0 and 10?")
            return

        # add the data to the pandas dataframe
        datapoint = pd.Series({'cook_seconds': time,
                        'weight_gr': weight,
                        'rating': rating})
        self.egg_data = pd.concat([self.egg_data, datapoint.to_frame().T])

        self.egg_data = self.egg_data.drop_duplicates()

        # save the data to the csv file
        # TODO: write this to a file in a Docker volume
        self.egg_data.to_csv(self.data_file, index=False, sep=",")

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
        self.output_message(msg)


def create_skill():
    return SoggyEggTimerSkill()
