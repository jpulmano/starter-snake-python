import numpy as np
import random

class MyBattlesnakeHeuristics:
    '''
    The BattlesnakeHeuristics class allows you to define handcrafted rules of the snake.
    '''
    
    FOOD_INDEX = 0
    def __init__(self, json):
        self.height = json["board"]["height"]
        self.width = json["board"]["width"]
        self.foods = json["board"]["food"]
        self.snakes = json["board"]["snakes"]
        self.me = json["you"]
        self.my_health = json["you"]["health"]
        self.my_body = json["you"]["body"]
        self.my_head = json["you"]["head"]
        self.my_length = json["you"]["length"]
    
    
    def go_to_food_if_close(self):
        '''
        Example heuristic to move towards food if it's close to you.
        '''
        
        # Get the position of the snake head
        i_head, j_head = self.my_head["x"], self.my_head["y"]
        UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
        
        # Check if we are surrounded by food
        if {'x': i_head-1, 'y': j_head} in self.foods:
            food_direction = LEFT
        if {'x': i_head+1, 'y': j_head} in self.foods:
            food_direction = RIGHT
        if {'x': i_head, 'y': j_head-1} in self.foods:
            food_direction = DOWN
        if {'x': i_head, 'y': j_head+1} in self.foods:
            food_direction = UP
        
        return food_direction
    
    def did_try_to_kill_self(self, action):
        
        # Return if our body is smol (1 piece only)
        if len(self.my_body) == 1:
            return False
        
        i_head, j_head = self.my_head["x"], self.my_head["y"]
        
        # Get the position of the second piece (body)
        i_body, j_body = self.my_body[1]["x"], self.my_body[1]["y"]

        # Calculate the facing direction with the head and the next location
        diff_horiz, diff_vert = i_head - i_body, j_head - j_body
        UP, DOWN, LEFT, RIGHT = 0, 1, 2 ,3
        
        if diff_horiz == -1 and diff_vert == 0: # Left
            if action == RIGHT:
                return True
        elif diff_horiz == 1 and diff_vert == 0: # Right
            if action == LEFT:
                return True 
        elif diff_horiz == 0 and diff_vert == 1: # Up
            if action == DOWN:
                return True
        elif diff_horiz == 0 and diff_vert == -1: # Down
            if action == UP:
                return True
            
        return False

    def did_try_to_escape(self, action):

        # Get the position of snake head
        i_head, j_head = self.my_head["x"], self.my_head["y"]

        # Get dimensions
        height_min, width_min, height_max, width_max = 0, 0, self.height-1, self.width-1

        UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

        # Top, bottom, left, right layer respectively
        if j_head == height_min and action == DOWN \
            or j_head == height_max and action == UP \
            or i_head == width_min and action == LEFT \
            or i_head == width_max and action == RIGHT:
            return True

        return False
    
    def did_try_to_hit_snake(self, action):
        # Get the position of snake head
        i_head, j_head = self.my_head["x"], self.my_head["y"]
        
        # Compute the next location of snake head with action
        UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
        if action == UP:
        	j_head += 1
       	elif action == DOWN:
       		j_head -= 1
       	elif action == LEFT:
       		i_head -= 1
       	elif action == RIGHT:
       		i_head += 1
         
        # Loop through snakes to see if we're about to collide
        i = 0
        for snake in self.snakes:
            i += 1
            if i == 1: # Ignore our own body for now
                continue
            for piece in snake["body"]:
                x, y = piece["x"], piece["y"]
                if x == i_head or y == j_head:
                    return True

        return False
    
    def run(self):
        
        log_string = ""
        
        actions = np.random.rand(4) # Walmart Q-values
        action = int(np.argmax(actions))
    
        num_redirected = 2 # If we ignore action, choose second highest and continue decrementing.
        
        # Example heuristics to eat food that you are close to.
        if self.my_health < 50:
            food_direction = self.go_to_food_if_close()
            if food_direction:
                action = food_direction
                log_string = "Went to food if close."
        
        # Don't do a forbidden move
        if self.did_try_to_kill_self(action):
            
            old_action = action # For logging
            
            # Pick one lower q-value instead
            sort = np.argsort(actions)
            action = sort[-num_redirected]

            num_redirected -= 1
            
            log_string = "Forbidden. Changed {} to {}".format(old_action, action)

        # Don't exit the map
        if self.did_try_to_escape(action):

            old_action = action # For logging
            
            # Pick one lower q-value instead
            sort = np.argsort(actions)
            action = sort[-num_redirected]

            num_redirected -= 1
            
            log_string = "Tried to escape. Changed {} to {}".format(old_action, action)

        # Don't hit another snake
        if self.did_try_to_hit_snake(action):
            old_action = action
            
            # Pick one lower q-value instead
            sort = np.argsort(actions)
            action = sort[-num_redirected]
            
            num_redirected -= 1
            
            log_string = "About to hit a snake. Changed {} to {}".format(old_action, action)

        # Somehow, every action is bad! Pick the highest q-value and die lmao
        if action not in [0, 1, 2, 3]:
        	action = int(np.argmax(action))
        	log_string = "Guess I'll die!"

        assert action in [0, 1, 2, 3], "{} is not a valid action.".format(action)
        
        return action, log_string