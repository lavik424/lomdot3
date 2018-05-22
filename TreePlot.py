
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
import sklearn.datasets as datasets
import pandas as pd


def plotTree(dtree:DecisionTreeClassifier, features_names, classes_names):
    dot_data = StringIO()
    # export_graphviz(dtree, out_file=dot_data)
    #                 # filled=True, rounded=True,
    #                 # special_characters=True)


    export_graphviz(dtree, out_file=dot_data,
                    feature_names=features_names,
                    class_names=classes_names,
                    filled=True, rounded=True,
                    special_characters=True)

    print(dot_data.getvalue())

    graph = graphviz.Source(dot_data.getvalue())

    graph.render(view=True, cleanup=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())


if __name__ == '__main__':
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    dtree = DecisionTreeClassifier()
    dtree.fit(df, y)
    plotTree(dtree, iris.feature_names, iris.target_names)