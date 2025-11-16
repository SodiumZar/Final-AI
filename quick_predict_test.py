import numpy as np, cv2, os
from predict import predict

# Create a synthetic retina-like image for testing
h, w = 512, 512
img = np.zeros((h,w,3), dtype=np.uint8)
cv2.circle(img, (w//2,h//2), 230, (30,60,30), -1)  # dark greenish background
# draw some vessel-like lines
for i in range(0, 360, 15):
    x = int(w/2 + 200*np.cos(np.deg2rad(i)))
    y = int(h/2 + 200*np.sin(np.deg2rad(i)))
    cv2.line(img, (w//2,h//2), (x,y), (10,20,10), 2)

os.makedirs('static/uploads', exist_ok=True)
path = 'static/uploads/synthetic.jpg'
cv2.imwrite(path, img)

res = predict(path)
if isinstance(res, dict):
    mask = res['mask']
    print('Fallback used, mask shape:', mask.shape, 'unique:', np.unique(mask))
else:
    mask = res
    print('Model used, mask shape:', mask.shape, 'unique:', np.unique(mask))
