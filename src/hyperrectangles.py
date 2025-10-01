from sentence_transformers import util
import pickle as pk
import numpy as np
import os


def contained(point, hyperrectangle):
    for i in range(len(point)):
        if point[i] < hyperrectangle[i][0] or point[i] > hyperrectangle[i][1]:
            return False
    return True


def calculate_hyperrectangle(points):
    # Calculate the hyperrectangle around the points
    min_max_list = np.full((points.shape[1], 2), [np.inf, -np.inf])
    for point in points:
        for i in range(len(point)):
            if point[i] < min_max_list[i][0]:
                min_max_list[i][0] = point[i]
            if point[i] > min_max_list[i][1]:
                min_max_list[i][1] = point[i]
    return min_max_list



def load_hyperrectangles(X_pos_train_embed_align, X_pos_train_PCA, Xp_pos_train_embed_align, Xp_pos_train_PCA, train_pos_index):

    hyperrectangles = []
    for i in range(len(X_pos_train_PCA)):
        points = []
        points.append(X_pos_train_PCA[i])
        # print("points length:", len(points))
        for index, value in enumerate(train_pos_index):
            if value == i:
                cosine_score = util.cos_sim(X_pos_train_embed_align[i], Xp_pos_train_embed_align[index])
                #print("cosine score:", cosine_score)
                if cosine_score > 0.6:
                    points.append(Xp_pos_train_PCA[index])
        if len(points) >= 2:
            hyperrectangle = calculate_hyperrectangle(np.array(points))
            hyperrectangles.append(hyperrectangle)

    print(np.array(hyperrectangles).shape)
    return hyperrectangles
