from reelify.app import app
app.config["JSON_AS_ASCII"] = False
if __name__ == "__main__":
    app.run()
