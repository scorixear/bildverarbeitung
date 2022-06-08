# Wesentliche Unterschiede RGB vs HSV
# Darstellung der Farben bei RGB durch RGB-Anteile (additive Farberzeugung)
# HSV: Farbdarstellung durch HUE-Wert als Interpolation im Farbkreis
# und Saturation (Sättigung = Grayscale) / Value (Helligkeit)
# Bei RGB Informationsverlust bei Verdunklung der Farbe, bei HSV nicht

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# RGB Color Interpolation
def colorFaderRGB(color1, color2, mix=0):
  color1 = np.array(mpl.colors.to_rgb(color1))
  color2 = np.array(mpl.colors.to_rgb(color2))
  return (1-mix)*color1 + mix*color2;


# HSV Color Interpolation
def colorFaderHSV(color1, color2, mix=0):
  color1 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color1)));
  color2 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color2)));
  if(color1[0] < color2[0]):
    if color2[0] - color1[0] < 180:
      mix = 1-mix;
  else:
    if color1[0]-color2[0] > 180:
      mix = 1-mix;

  return mpl.colors.hsv_to_rgb((1-mix)*color1 + mix*color2)


color1 = "blue"
color2 = "red"
n = 100
img = np.zeros((50,n,3))
for x in range(50):
  for y in range(n):
    img[x,y]= colorFaderHSV(color1, color2, y/n)
plt.imshow(img)
plt.show()



