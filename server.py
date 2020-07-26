import os
import random
import cherrypy
import time
import json
import torch

from heuristics import Heuristics
from model import make_agent
from generator import GameGenerator

class Battlesnake(object):
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        # This function is called when you register your Battlesnake on play.battlesnake.com
        # It controls your Battlesnake appearance and author permissions.
        # TIP: If you open your Battlesnake URL in browser you should see this data
        return {
            "apiversion": "1",
            "author": "jpulmano",
            "color": "#ff8f00", # Princeton orange bih
            "head": "bendr",
            "tail": "freckled",
        }

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def start(self):
        # This function is called everytime your snake is entered into a game.
        # cherrypy.request.json contains information about the game that's about to be played.
        # TODO: Use this function to decide how your snake is going to look on the board.
        data = cherrypy.request.json
        
        # Create a model
        # agent = make_agent()
        # print('AGENT', agent)

        print("START")
        return "ok"

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def move(self):
        # This function is called on every turn of a game. It's how your snake decides where to move.
        # Valid moves are "up", "down", "left", or "right".

        json = cherrypy.request.json
        possible_moves = ["up", "down", "left", "right"]

        # ---------------------------------------- 
        
        # Choose an action through heuristics
        # heuristics = Heuristics(json)
        # action_index, log_strings = heuristics.run()
        
        # ---------------------------------------- 
        
        # Create an agent
        agent, policy = make_agent()
    
        device = torch.device('cpu')
    
        # (Old) Get the action our policy should take
        # _, action, _, _ = policy.act(torch.tensor(100, dtype=torch.float32).to(device), None, None)
    
        # Set up the game generator
        layers = 17
        height = json["board"]["height"]
        width = json["board"]["width"]
        gen = GameGenerator(layers, height, width)
        
        # Convert the json
        agent_input = torch.tensor(gen.make_input(json), dtype=torch.float32)
        
        # Get the action
        start = time.time()
        with torch.no_grad():
            action, value = agent.predict(agent_input, deterministic=True)
        end = time.time()

        print(action)
        
        # Print move
        print("Step {}... Move: {}".format(json['turn'], action))
        print("Score: {} calculated in {} seconds".format(value[0].item(), end-start))
        
        return {"move": action}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def end(self):
        # This function is called when a game your snake was in ends.
        # It's purely for informational purposes, you don't have to make any decisions here.
        data = cherrypy.request.json

        print("END")
        
        if data["you"] not in data["board"]["snakes"]:
            print("you lost!")
        else:
            print("you won!")
            
        return "ok"


if __name__ == "__main__":
    server = Battlesnake()
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.config.update(
        {"server.socket_port": int(os.environ.get("PORT", "8080")),}
    )
    print("Starting Battlesnake Server...")
    cherrypy.quickstart(server)
