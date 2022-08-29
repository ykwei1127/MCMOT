#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import os
import sys
import zipfile
import tarfile
import traceback
import numpy as np
import pandas as pd
import scipy as sp
import motmetrics as mm
import pytrec_eval as trec
from PIL import Image
from collections import defaultdict
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("data", nargs=2, help="Path to <test_labels> <predicted_labels>.")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument('-m', '--mread', action='store_true', help="Print machine readable results (JSON).")
    parser.add_argument('-ds', '--dstype', type=str, default='train', help="Data set type: train, validation or test.")
    return parser.parse_args()


def usageMsg():
    return """  python3 eval.py <ground_truth> <prediction> --dstype <dstype>

Details for expected formats can be found at https://www.aicitychallenge.org/.

See `python3 eval.py --help` for more info.

"""


def getData(fh, fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    
    Params
    ------
    fh : opened handle
        Steam handle to read from.
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """
    
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            index_col=None, 
            skipinitialspace=True, 
            header=None,
            names=names,
            engine='python'
        )
        
        return df
    
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def readData(fpath):
    """ Read test or pred data for a given track. 
    
    Params
    ------
    fpath : str
        Original path of file reading from.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    Exceptions
    ----------
        May raise a ValueError exception if file cannot be opened or read.
    """
    names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
        
    if not os.path.isfile(fpath):
        raise ValueError("File %s does not exist." % fpath)
    # Gzip tar archive
    if fpath.lower().endswith("tar.gz") or fpath.lower().endswith("tgz"):
        tar = tarfile.open(fpath, "r:gz")
        members = tar.getmembers()
        if len(members) > 1:
            raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
        if not members:
            raise ValueError("Missing files in archive %s." % fpath)
        fh = tar.extractfile(members[0])
        return getData(fh, tar.getnames()[0], names=names)
    # Zip archive
    elif fpath.lower().endswith(".zip"):
        with zipfile.ZipFile(fpath) as z:
            members = z.namelist()
            if len(members) > 1:
                raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
            if not members:
                raise ValueError("Missing files in archive %s." % fpath)
            with z.open(members[0]) as fh:
                return getData(fh, members[0], names=names)
    # text file
    elif fpath.lower().endswith(".txt"):
        with open(fpath, "r") as fh:
            return getData(fh, fpath, names=names)
    else:
        raise ValueError("Invalid file type %s." % fpath)


def print_results(summary, mread=False):
    """Print a summary dataframe in a human- or machine-readable format.
    
    Params
    ------
    summary : pandas.DataFrame
        Data frame of evaluation results in motmetrics format.
    mread : bool
        Whether to print results in machine-readable format (JSON).
    Returns
    -------
    None
        Prints results to screen.
    """
    if mread:
        print('{"results":%s}' % summary.iloc[-1].to_json())
        return
    
    formatters = {'idf1': '{:2.2f}'.format,
    		      'mota': '{:2.2f}'.format,
                  'motp': '{:2.2f}'.format,
                  'num_unique_objects': '{:2.2f}'.format,
                  'mostly_tracked': '{:2.2f}'.format,
                  'mostly_lost': '{:2.2f}'.format,
                  'precision': '{:2.2f}'.format,
                  'recall': '{:2.2f}'.format}
    
    summary = summary[['idf1', 'idtp', 'idfp', 'idfn', 'mota', 'motp','num_unique_objects','mostly_tracked', 'mostly_lost', 'precision', 'recall', 'num_switches']]
    summary['idf1'] *= 100
    summary['mota'] *= 100
    summary['motp']  = (1 - summary['motp']) * 100
    summary['precision'] *= 100
    summary['recall'] *= 100
    summary['mostly_tracked'] = summary['mostly_tracked'] / summary['num_unique_objects'] * 100
    summary['mostly_lost'] = summary['mostly_lost'] / summary['num_unique_objects'] * 100

    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))
    return


def eval(test, pred, **kwargs):
    """ Evaluate submission.

    Params
    ------
    test : pandas.DataFrame
        Labeled data for the test set. Minimum columns that should be present in the 
        data frame include ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height'].
    pred : pandas.DataFrame
        Predictions for the same frames as in the test data.
    Kwargs
    ------
    mread : bool
        Whether printed result should be machine readable (JSON). Defaults to False.
    dstype : str
        Data set type. One of 'train', 'validation' or 'test'. Defaults to 'train'.
    Returns
    -------
    df : pandas.DataFrame
        Results from the evaluation
    """
    if test is None:
        return None
    mread  = kwargs.pop('mread', False)
    dstype = kwargs.pop('dstype', 'train')
    
    def removeRepetition(df):
        """Remove repetition to ensure that all objects are unique for every frame.

        Params
        ------
        df : pandas.DataFrame
            Data that should be filtered
        Returns
        -------
        df : pandas.DataFrame
            Filtered data that all objects are unique for every frame.
        """

        df = df.drop_duplicates(subset=['CameraId', 'Id', 'FrameId'], keep='first')

        return df
        
    def compare_dataframes_mtmc(gts, ts):
        """Compute ID-based evaluation metrics for multi-camera multi-object tracking.
        
        Params
        ------
        gts : pandas.DataFrame
            Ground truth data.
        ts : pandas.DataFrame
            Prediction/test data.
        Returns
        -------
        df : pandas.DataFrame
            Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
            With more columns: 'mota', 'motp', 'mostly_tracked', 'mostly_lost', 'precision', 'recall', 'num_switches'
        """
        gtds = []
        tsds = []
        gtcams = gts['CameraId'].drop_duplicates().tolist()
        tscams = ts['CameraId'].drop_duplicates().tolist()
        maxFrameId = 0;

        for k in sorted(gtcams):
            gtd = gts.query('CameraId == %d' % k)
            gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId in gtd only
            mfid = gtd['FrameId'].max()
            gtd['FrameId'] += maxFrameId
            gtd = gtd.set_index(['FrameId', 'Id'])
            gtds.append(gtd)

            if k in tscams:
                tsd = ts.query('CameraId == %d' % k)
                tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
                # max FrameId among both gtd and tsd
                mfid = max(mfid, tsd['FrameId'].max())
                tsd['FrameId'] += maxFrameId
                tsd = tsd.set_index(['FrameId', 'Id'])
                tsds.append(tsd)

            maxFrameId += mfid

        # compute multi-camera tracking evaluation stats
        multiCamAcc = mm.utils.compare_to_groundtruth(pd.concat(gtds), pd.concat(tsds), 'iou')
        metrics=list(mm.metrics.motchallenge_metrics)
        metrics.extend(['idf1','idfp','idfn','idtp','mota','motp','num_unique_objects','mostly_tracked','mostly_lost','precision','recall','num_switches'])
        summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')

        return summary

    mh = mm.metrics.create()
    
    # filter prediction data
    pred = removeRepetition(pred)
    
    # evaluate results
    return compare_dataframes_mtmc(test, pred)


def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()


if __name__ == '__main__':
    args = get_args();
    if not args.data or len(args.data) < 2:
        usage("Incorrect number of arguments. Must provide paths for the test (ground truth) and predicitons.")
    
    test = readData(args.data[0])
    pred = readData(args.data[1])
    try:
        summary = eval(test, pred, mread=args.mread, dstype=args.dstype)
        print_results(summary, mread=args.mread)
    except Exception as e:
        if args.mread:
            print('{"error": "%s"}' % repr(e))
        else: 
            print("Error: %s" % repr(e))
        traceback.print_exc()
