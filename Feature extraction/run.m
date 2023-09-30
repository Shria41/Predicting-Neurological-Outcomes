%run a for loop through training files, call load_recording->save 3 things,
%call get features function->store features in a csv file (first print it)

input_directory="D:\Capstone\i-care-international-cardiac-arrest-research-consortium-database-1.0\training";
files = dir (input_directory);
L = length (files);
for i=1:L
   
   
   patient_id=convertCharsToStrings(files(i).name);
   disp(patient_id);
   %disp(class(patient_id));
   if(startsWith(patient_id,"ICARE"))
       
       [patient_metadata,recording_metadata,recordings]=load_challenge_data(input_directory,patient_id);
       
       features=get_features(patient_metadata, recording_metadata, recordings);
       
       dlmwrite("features_band.csv",features,'-append');
       %disp(features(15));
   end
end


%patient_id='ICARE_0284';


%patient_metadata_file=fullfile(input_directory,patient_id,[strcat(patient_id,".txt")]);
%patient_metadata=fileread(patient_metadata_file);


%loop through all the directories and cpc score is stored where?

