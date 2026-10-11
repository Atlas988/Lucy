def style_response(response, sentiment):
    if sentiment == 'positive':
        return response + " 😊"
    elif sentiment == 'neutral':
        return response
    elif sentiment == 'negative':
        return response + " 😟"
    else:
        return response
