# OpenFermion Cloud Library (Alpha)

The OpenFermion team provides an open source data repository where users can
create and share molecular data files created in OpenFermion. The idea is that
these files may serve as benchmarks for the field and may also provide easy
access to different molecules for users who are not comfortable using our
electronic structure package plugins. Eventually, this library might grow to
include other types of data such as measurements from a relevant experiment.


## How do I download these data files?

The text files in this folder are manifests for currently available datasets on
the cloud. Each manifest contains a short description of the data and makes it
clear how files are named. The remainer of the manifest lists the name of
available data files. You can find and download files you want by
clicking on them in our [molecule cloud
website](https://quantumlib.github.io/openfermioncloud/).


## How do I upload data files to share with other OpenFermion users?

For security reasons there is a somewhat elaborate protocol to contributing to
the database. We hope to simplify things in the future but the solution we have
decided on for now, after exploring many options (Github LFS, Google Drive,
simple web app, etc.), is to build a system with Google Cloud Storage and to use
Github and Google Cloud Signed URL for source control. To contribute, follow the
following steps:

1. To create a new dataset (or contribute to an existing dataset), first
   generate the files you intend to upload and compress them into a single
   file. If you use something other than .zip, say what you used in your pull
   request.

2. Open a pull request on OpenFermion which creates a new file in this folder
   (if creating a new dataset) or edits an existing file in this folder (if
   expanding an existing dataset). The file should be named after the dataset.
   The first line of the file should describe the dataset and subsequent lines
   should contain the names of the files in the dataset. See existing files in
   the folder for an example. Be sure to have signed the CLA!

3. Either give your email in the pull request or email us so we can privately
   send you a signed URL for the upload.

4. You will recieve an email with a signed URL which you should use to upload
   your data to our staging bucket, For instance, you can use the command
   ```
   curl -X PUT --upload-file <your_files.zip> <signed_url>
   ```

5. Wait for OpenFermion administrators to perform a security review of your data
   and then transfer your data to the public facing producting bucket from where
   it can be downloaded. We will signal that your files have been exposed to
   the public facing cloud library by merging your pull request.
