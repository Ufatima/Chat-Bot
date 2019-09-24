from flask import Flask, render_template, request

from ChatInstance import ChatInstance

app = Flask(__name__, static_url_path='')



Chatter = ChatInstance('Natural_language_generation\\model')


@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    out_txt = Chatter.converse(userText)
    return str(out_txt)
    
@app.route("/values")
def values():
    values_dict = Chatter.prev_conversation[-1]
    if not values_dict['actor'] == 'user':
        values_dict = Chatter.prev_conversation[-2]
    template = render_template("values.html", emotion=str(values_dict['emotions']),
                            dialogue=str(values_dict['da']),
                            topic=str(values_dict['topic']),
                            extra=str(values_dict['personality'][0]),
                            stable=str(values_dict['personality'][1]),
                            agreeable=str(values_dict['personality'][2]),
                            conscient=str(values_dict['personality'][3]),
                            open=str(values_dict['personality'][4]))
    return template


if __name__ == "__main__":
    app.run(port=5002,debug=True)


    
