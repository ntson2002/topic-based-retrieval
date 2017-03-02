import web
import json

urls = ('/', 'index')

class index:
	def GET(self):
		web.header('Content-Type', 'application/json')
		dic = {'a':5, 'b':5}
		return json.dumps(dic, indent=4, sort_keys=True)

if __name__ == '__main__':
	app = web.application(urls, globals())
	app.run()

