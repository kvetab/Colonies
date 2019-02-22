import LoadImage


loader = LoadImage.ImageLoader('colony4b.csv',  'empty4b.csv')
loader.openImage()
loader.SaveToFile(True)
loader.openImage()
loader.SaveToFile(False)




