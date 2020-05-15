# Use this carefully!!!
svn status | grep '^!' | awk '{ print $2}' | xargs svn delete
