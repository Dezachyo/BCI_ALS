using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LSL;

namespace LSL4Unity.Samples.SimpleInlet
{ 
    // You probably don't need this namespace. We do it to avoid contaminating the global namespace of your project.
    public class LSL_Receiver : MonoBehaviour
    {

        // We need to find the stream somehow. You must provide a StreamName in editor or before this object is Started.
        public string StreamName;
        ContinuousResolver resolver;

        double max_chunk_duration = 0.2;  // Duration, in seconds, of buffer passed to pull_chunk. This must be > than average frame interval.

        // We need to keep track of the inlet once it is resolved.
        private StreamInlet inlet;

        // We need buffers to pass to LSL when pulling data.
        private float[,] data_buffer;  // Note it's a 2D Array, not array of arrays. Each element has to be indexed specifically, no frames/columns.
        private double[] timestamp_buffer;
        private double timestamp;
        private string[] sample;

        public GameObject Siren, Background, TextGameobject;
        public P300_Online p300_Online_script;


        void Start()
        // {
        //     if (!StreamName.Equals(StreamName))
        //         resolver = new ContinuousResolver("name", StreamName);
                
        //     else
        //     {
        //         Debug.LogError("Object must specify a name for resolver to lookup a stream.");
        //         this.enabled = false;
        //         return;
        //     }
        //     StartCoroutine(ResolveExpectedStream());
        // }
        {
            resolver = new ContinuousResolver("name", StreamName);
            StartCoroutine(ResolveExpectedStream());
        }
        IEnumerator ResolveExpectedStream()
        {

            var results = resolver.results();
            Debug.Log (results);
            while (results.Length == 0)
            {
                yield return new WaitForSeconds(.1f);
                results = resolver.results();
            }
            
            inlet = new StreamInlet(results[0]);
            Debug.Log (results[0]);
            // Prepare pull_chunk buffer
            int buf_samples = (int)Mathf.Ceil((float)(inlet.info().nominal_srate() * max_chunk_duration));
            // Debug.Log("Allocating buffers to receive " + buf_samples + " samples.");
            int n_channels = inlet.info().channel_count();
            data_buffer = new float[buf_samples, n_channels];
            timestamp_buffer = new double[buf_samples];
            Debug.Log(n_channels);
            sample = new string[n_channels];
        }

        // Update is called once per frame
        void Update()
        {
            if (inlet != null)
            {
                double samples_returned = inlet.pull_sample(sample, timestamp);
                // Debug.Log("Samples returned: " + samples_returned);
                if (samples_returned > 0)
                {
                    Debug.Log(sample[0]);
                    if (sample[0] == "Y")
                    {
                        // Emergency signal variation
                        Siren.SetActive(true);
                        Background.SetActive(false);
                        TextGameobject.SetActive(false);

                        // Simple Yes/No variation 
                        // p300_Online_script.textText.color = new Color(0, 255, 0, 255);
                        // p300_Online_script.textText.text = "You Were Thinking - Yes";
                    }
                    else if (sample[0] == "N")
                    {
                        // Emergency signal variation
                        p300_Online_script.textText.color = new Color(0, 255, 0, 255);
                        p300_Online_script.textText.text = "No Emergency - הכל טוב";

                        // Simple Yes/No variation 
                        // p300_Online_script.textText.color = new Color(255, 0, 0, 255);
                        // p300_Online_script.textText.text = "You Were Thinking - No";
                    }
                    // There are many things you can do with the incoming chunk to make it more palatable for Unity.
                    // Note that if you are going to do significant processing and feature extraction on your signal,
                    // it makes much more sense to do that in an external process then have that process output its
                    // result to yet another stream that you capture in Unity.
                    
                }
            }
        }
    }
}