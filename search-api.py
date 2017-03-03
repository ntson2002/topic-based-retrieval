import web
import json

import query as api

urls = ('/users', 'list_users',
        '/api/search/(.*)', 'api_query',
        '/api/f=topics', 'api_topics',
        '/api/search2/(.*)/topic/(.+)', 'api_query_topic')


class list_users:
	def GET(self):
		web.header('Content-Type', 'application/json')
		dic = {'a':5, 'b':5, 'd':l[2]}
		return json.dumps(dic, indent=4, sort_keys=True)

class api_topics:
    def GET(self):
        web.header('Content-Type', 'application/json')
        data = api.getTopics()
        return json.dumps(data, indent=4, sort_keys=True)

class api_query:
    def GET(self, q):
        print "function: api_query"
        print "q:", q
        web.header('Content-Type', 'application/json')
        model = "output/model_TFIDF.pkl"
        data = api.queryTFIDF(model, q, 20)
        return json.dumps(data, indent=4, sort_keys=True)

class api_query_topic:
    def GET(self, q, t):
        print "function: api_query_topic"
        print "q:", q
        print "t:", t
        web.header('Content-Type', 'application/json')
        model = "output/model_TFIDF.pkl"
        topicFile = "output/topic.pickle"
        data = api.queryTFIDF_topicBased(model, topicFile, q, t, 30)
        return json.dumps(data, indent=4, sort_keys=True)

class SearchAPIApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

# if __name__ == "__main__":
#     app = SearchAPIApplication(urls, globals())
#     app.run(port=8081)


import optparse
if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-p", "--port", default="8081",
        type='int',
        help="port"
    )

    opts = optparser.parse_args()[0]
    app = SearchAPIApplication(urls, globals())
    print "API server started ... "
    app.run(port=opts.port)



