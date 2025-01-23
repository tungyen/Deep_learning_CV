import os

import torch
import open3d as o3d
import numpy as np

from pointNet import *

color_map = {
    0: [1, 0, 0],  # Class 0: Red
    1: [0, 1, 0],  # Class 1: Green
    2: [0, 0, 1],   # Class 2: Blue
    3: [1, 1, 0]
}

def pointNet_predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testPath = os.path.join("../..", 'Dataset', "Chair_dataset", 'test')
    testFiles = os.listdir(testPath)
    
    model = PointNetSegmentation()
    weightPath = "ckpts/pointNet.pth"
    model.load_state_dict(torch.load(weightPath, map_location=device))
    model = model.to(device)
    
    for testFile in testFiles:
        pcdPath = os.path.join(testPath, testFile)
        pcd = o3d.io.read_point_cloud(pcdPath, format='xyz')
        points = np.asarray(pcd.points)
        pointNP = points.copy()
        
        mu = np.mean(points, axis=0)
        var = np.mean(np.square(points-mu))
        points = (points-mu) / np.sqrt(var)
        points = torch.tensor(points, dtype=torch.float64).transpose(1, 0)
        points = torch.unsqueeze(points, 0)
        
        
        model.eval()
        with torch.no_grad():
            points = points.to(device).float()
            output = torch.squeeze(model(points))
            predict = torch.softmax(output, dim=0).cpu()
            predict_cla = torch.argmax(predict, dim=0).numpy()
            
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pointNP)
            colors = np.array([color_map[label] for label in predict_cla])
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([point_cloud])
    
    
if __name__ == '__main__':
    pointNet_predict()