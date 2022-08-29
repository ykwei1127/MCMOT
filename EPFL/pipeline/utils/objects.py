import math

class Track(object):

    def __init__(self):
        self.id = None
        self.feature_list = list()
        self.frame_list = list()
        self.box_list = list()
        self.su_list = list()
    
    def __len__(self):
        return len(self.frame_list)

    def sort(self):
        self.frame_list, self.feature_list, self.box_list, self.su_list = (list(t) for t in zip(*sorted(zip(self.frame_list, self.feature_list, self.box_list, self.su_list))))

class TrackOri(object):
    
    def __init__(self):
        self.id = None
        self.feature_list = list()
        self.frame_list = list()
        self.box_list = list()
    
    def __len__(self):
        return len(self.frame_list)

    def sort(self):
        self.frame_list, self.feature_list, self.box_list = (list(t) for t in zip(*sorted(zip(self.frame_list, self.feature_list, self.box_list))))

class GroupNode(object):
    def __init__(self, match_ids, id, thres):
        self.id = id
        self.match_ids = match_ids
        self.parent = None
        self.max_intersection = thres

    def __len__(self):
        return len(self.match_ids)

    def set_parent_id(self):
        parentnode = self.parent
        while parentnode.parent != None:
            parentnode = parentnode.parent
        self.id = parentnode.id