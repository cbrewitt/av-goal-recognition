try:
    from urllib.request import urlopen
    from urllib.error import URLError
except ImportError:
    from urllib2 import urlopen
    from urllib2 import URLError

f = urlopen('https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/osx-homebrew.yaml', timeout=15)
text = f.read()
f.close()
print(text)
