// Copied from
// https://cdn.rawgit.com/chrisfilo/zenodo.js/v0.1/zenodo.js
var getContent = function(url, acceptType, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    if (acceptType == 'application/json') {
    	xhr.responseType = 'json';
    }
    xhr.setRequestHeader('Accept', acceptType);
    xhr.onload = function() {
        var status = xhr.status;
        if (status === 200) {
            callback(null, xhr.response);
        } else {
            callback(status, xhr.response);
        }
    };
    xhr.send();
};

String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
};

function getZenodoIDFromTag(conceptRecID, tagName, callback) {
    getContent('https://zenodo.org/api/records/?q=conceptrecid:' + conceptRecID + '%20AND%20related.identifier:*github*' + tagName + '&all_versions&sort=-version',
               'application/json',
                function(err, data) {
                if (err !== null) {
                    callback(err, null);
                } else {
                    if (data.length == 0) {
                      callback('No records found for this tag and Zenodo ID', null);
                    } else if (data.length > 1) {
                      callback('Ambiguous numnber of records (more than one) found for this tag and Zenodo ID', null);
                    } else {
                      targetID = data[0].id
                      callback(null, targetID);
                    }
                }
    });
}

function getLatestIDFromconceptID(conceptRecID, callback) {
    getContent('https://zenodo.org/api/records/' + conceptRecID,
               'application/json',
                function(err, data) {
                if (err !== null) {
                    callback(err, null);
                } else {
                    targetID = data.id
                    callback(null, targetID);
                }
    });
}

function getCitation(recordID, style, callback) {
  style = typeof style !== 'undefined' ? style : 'vancouver-brackets-no-et-al';
	getContent('https://www.zenodo.org/api/records/' + recordID + '?style=' + style, 'text/x-bibliography',
     function(err, data) {
        if (err !== null) {
           callback(err, null);
        } else {
           callback(null, data);
        }
      });
}

function getDOI(recordID, callback) {
	getContent('https://www.zenodo.org/api/records/' + recordID, 'application/json',
     function(err, data) {
        if (err !== null) {
           callback(err, null);
        } else {
           var DOI = data.doi;
           callback(null, DOI);
        }
      });
}

function getBIBTEX(recordID, callback) {
	getContent('https://www.zenodo.org/api/records/' + recordID, 'text/x-bibtex',
     function(err, data) {
        if (err !== null) {
           callback(err, null);
        } else {
           callback(null, data);
        }
      });
}
