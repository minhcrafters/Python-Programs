from pytube import YouTube

video = YouTube("https://www.youtube.com/watch?v=RynU-GoEoe4")
stream = video.streams.first()
print(video.title)
print(video.description)