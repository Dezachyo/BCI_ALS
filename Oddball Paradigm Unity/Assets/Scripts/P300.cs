using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Linq;
using System.IO;
using LSL;


public class P300 : MonoBehaviour
{
    // Variables declaration
    [Header("Participant Details")]
    public string SubNum;

    [Header("Objects References")]
    public bool stimTimer = false;
    public bool ISITimer = false;
    Stopwatch stopWatch = new Stopwatch();

    [Header("Task Parameters")]
    private float stimulusPresentationDuration;
    private float ISI;
    public int numberOfBlocks = 4;
    public float oddballRatio = 0.2f;
    public int numberOfTrials = 160;
    // /public float taskDuration = 300f;
    public float breakDuration = 120f;

    public GameObject background;
    public GameObject Target;
    public GameObject Distractor;
    public GameObject NonTarget; 
    private Sprite TargetSprite, NonTargetSprite, DistractorSprite;
    public GameObject Stimuli;
    public GameObject instructions;
    public GameObject textGameobject;
    public GameObject timerGameobject;
    public GameObject eventSystem;
    public Image currentInstructions;
    public TextMeshProUGUI textText;
    public TextMeshProUGUI timerText;
    public Sprite instructions2;
    public Sprite breakInstructions;
    public Sprite endScreen;

    // Behind The Scenes
    string filename ="";
    private int currentTrial;
    private int currentBlock;
    private float RT;
    private int numberOfTargetTrials,numberOfDistractorTrials;
    private float numberOfTargetTrialsF,numberOfDistractorTrialsF;
    [Header("Behind The Scenes")]
    public bool instructionsShown = false;
    public bool spacePress = false;
    public bool userPressed = false;
    public bool breakTimeOn = false;
    public static int[] arr1,arr2,arr3;
    public static int[] zeroArr = {0,0,0,0,0};  
    public static int[] practiceArr = {0,0,0,0,0,1,0,0,0,1}; 
    int[] trialTypeList;
    Dictionary<string, string> userResps = new Dictionary<string, string>();
    public float timeValue=0;


    List<int> Digits = new List<int>();

    [Header("LSL")] 
    string StreamName = "LSL4Unity.Samples.SimpleCollisionEvent";
    string StreamType = "Markers";
    private StreamOutlet outlet;
    private string[] sample = {""};

    
    // Start is called before the first frame update
    void Start()
    {
        //LSL
        var hash = new Hash128();
        hash.Append(StreamName);
        hash.Append(StreamType);
        hash.Append(gameObject.GetInstanceID());
        StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, 1, LSL.LSL.IRREGULAR_RATE,
            channel_format_t.cf_string, hash.ToString());
        outlet = new StreamOutlet(streamInfo);

        stimulusPresentationDuration = UnityEngine.Random.Range(0.3f, 0.5f); //Jitter
        ISI = UnityEngine.Random.Range(0.2f, 0.4f); //Jitter
        //Task
        currentTrial = 0; // Trial Counter

        textText = textGameobject.GetComponent<TextMeshProUGUI>(); // Text component for ongoing task, such as fixation point
        timerText = timerGameobject.GetComponent<TextMeshProUGUI>(); // Text component for onscreen timer
        currentInstructions = instructions.GetComponent<Image>(); // Current instruction shown component

        TargetSprite = Target.GetComponent<SpriteRenderer>().sprite;
        NonTargetSprite = NonTarget.GetComponent<SpriteRenderer>().sprite;
        DistractorSprite = Distractor.GetComponent<SpriteRenderer>().sprite;

        numberOfTargetTrialsF = oddballRatio * numberOfTrials; // Proportion of target 'oddball' trials, float
        numberOfDistractorTrialsF = (1-2*oddballRatio) * numberOfTrials; // Proportion of Distractor Trials, float
        numberOfTargetTrials = (int) numberOfTargetTrialsF; // Proportion of target 'oddball' trials, int
        numberOfDistractorTrials = (int) numberOfDistractorTrialsF; // Proportion of Distractor Trials, int
        arr1 = Enumerable.Repeat(1, numberOfTargetTrials).ToArray(); // Create target trials location array
        arr2 = Enumerable.Repeat(2, numberOfTargetTrials).ToArray(); // Create non-target trials location array
        arr3 = Enumerable.Repeat(0, (numberOfDistractorTrials-5)).ToArray(); // Create distractor trials location array
        trialTypeList = arr1.Concat(arr2).Concat(arr3).ToArray(); // Conjoin trial location arrays
        reshuffle(trialTypeList); // Shuffle trial locations
        trialTypeList = zeroArr.Concat(trialTypeList).ToArray(); // Add n Distractor Trials at the beginning of the experiment

        UnityEngine.Debug.Log("Target trials: " + numberOfTargetTrials + ", Non-Target trials: " + numberOfTargetTrials + ", Distractor trials: " +numberOfDistractorTrials);
        //filename = Application.dataPath + "/NBack_Subject_"+ SubNum + "_Block" + currentBlock + ".csv";
        StartCoroutine(P300Task());
    }   

    // Update is called once per frame
    void FixedUpdate()
    {
        if (stimTimer)
            if (stimulusPresentationDuration <= stopWatch.Elapsed.TotalSeconds)
                stimTimer = false;
        if (ISITimer)
            if (ISI <= stopWatch.Elapsed.TotalSeconds )
                ISITimer = false;
    }

    void TimerISI()
    {
        
    }

    void TimerStim()
    {
        
        stopWatch.Start();
        
    }
    

    void StopwatchUsingMethod()
    {
    //A: Setup and stuff you don't want timed
    var timer = new Stopwatch();
    bool execute = false;
    timer.Start();

    //B: Run stuff you want timed
    timer.Stop();
    execute = true;

    TimeSpan timeTaken = timer.Elapsed;
    string foo = "Time taken: " + timeTaken.ToString(@"m\:ss\.fff"); 
    }

    void DisplayTime (float timeToDisplay)
    {
        if (timeToDisplay <0)
        {
            timeToDisplay = 0;
        }

        float minutes = Mathf.FloorToInt(timeToDisplay / 60);
        float secondes = Mathf.FloorToInt(timeToDisplay % 60);

        timerText.text = string.Format("{0:00}:{1:00}", minutes, secondes);
    }
    void reshuffle(int[] trialTypesList)
    {
        // Knuth shuffle algorithm :: courtesy of Wikipedia :)
        for (int t = 0; t < trialTypesList.Length; t++ )
        {
            int tmp = trialTypesList[t];
            int r = UnityEngine.Random.Range(t, trialTypesList.Length);
            trialTypesList[t] = trialTypesList[r];
            trialTypesList[r] = tmp;
        }
    }

    public void WriteCSV()
    {
        TextWriter tw = new StreamWriter(filename, false);
        tw.WriteLine("TrialType,UserResponse");
        // for (int i = 1; i<=currentTrial; i++)
        // {
        //     tw.WriteLine(trialTypeList[i].ToString() + "," + userResps.ElementAt(i).Value) ;
        // }
        tw.Close();
    }

    IEnumerator P300Task()
            {
                // Start Task
                if (!instructionsShown) 
                {
                instructions.SetActive(true);
                while (!spacePress)
                {
                    if (Input.GetMouseButtonDown(0))
                    {
                        spacePress=true;
                    }
                    yield return null;
                }
                }
                spacePress=false;
                instructions.SetActive(false);
                background.SetActive(true);
                textGameobject.SetActive(true);
                for (int i = 0; i < numberOfBlocks; i++)
                {
                    ResetParameters();
                    textText.text = "ליחתהל תדמוע הלטמה";
                    yield return new WaitForSecondsRealtime(1f);
                    textText.text = "3";
                    yield return new WaitForSecondsRealtime(1f);
                    textText.text = "2";
                    yield return new WaitForSecondsRealtime(1f);
                    textText.text = "1";
                    yield return new WaitForSecondsRealtime(1f);
                    textText.text = "+"; // Crosshair first+
                    yield return new WaitForSecondsRealtime(0.25f); // ISI
                    foreach (int trialType in trialTypeList)
                        { 
                            stimulusPresentationDuration = UnityEngine.Random.Range(0.3f, 0.5f); //Jitter
                            ISI = UnityEngine.Random.Range(0.2f, 0.4f); //Jitter
                            currentTrial++;
                            UnityEngine.Debug.Log ("current trial " + currentTrial);
                            userPressed = false;
                            if (currentTrial<(5))
                            {  
                                textGameobject.SetActive(false);
                                Stimuli.GetComponent<Image>().sprite = DistractorSprite ;
                                Stimuli.SetActive(true);
                                if (outlet != null)
                                    {
                                        sample[0] = "Distractor Trial";
                                        UnityEngine.Debug.Log(sample[0]);
                                        outlet.push_sample(sample);
                                    }
                                stimTimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (stimTimer)
                                {
                                    yield return null;
                                }
                                textGameobject.SetActive(true);
                                Stimuli.SetActive(false);
                                ISITimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (ISITimer)
                                {
                                    yield return null;
                                }
                            }
                            else 
                            {
                            if (trialType == 1)
                                {
                                textGameobject.SetActive(false);   
                                Stimuli.GetComponent<Image>().sprite = TargetSprite ;
                                Stimuli.SetActive(true);
                                if (outlet != null)
                                    {
                                        sample[0] = "Target Trial";
                                        UnityEngine.Debug.Log(sample[0]);
                                        outlet.push_sample(sample);
                                    }
                                stimTimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (stimTimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(stimulusPresentationDuration);
                                textGameobject.SetActive(true);
                                Stimuli.SetActive(false);
                                ISITimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (ISITimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(ISI);// ISI
                                }
                            if (trialType == 2)
                                {
                                textGameobject.SetActive(false);   
                                Stimuli.GetComponent<Image>().sprite = NonTargetSprite ;
                                Stimuli.SetActive(true);
                                if (outlet != null)
                                    {
                                        sample[0] = "Non-Target Trial";
                                        UnityEngine.Debug.Log(sample[0]);
                                        outlet.push_sample(sample);
                                    }
                                stimTimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (stimTimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(stimulusPresentationDuration);
                                textGameobject.SetActive(true);
                                Stimuli.SetActive(false);
                                ISITimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (ISITimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(ISI);// ISI
                                }
                            if (trialType == 0)
                                {
                                textGameobject.SetActive(false);
                                Stimuli.GetComponent<Image>().sprite = DistractorSprite ;
                                Stimuli.SetActive(true);
                                if (outlet != null)
                                    {
                                        sample[0] = "Distractor Trial";
                                        UnityEngine.Debug.Log(sample[0]);
                                        outlet.push_sample(sample);
                                    }
                                stimTimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (stimTimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(stimulusPresentationDuration);
                                textGameobject.SetActive(true);
                                Stimuli.SetActive(false);
                                ISITimer = true;
                                stopWatch.Reset();
                                stopWatch.Start();
                                while (ISITimer)
                                {
                                    yield return null;
                                }
                                //yield return new WaitForSecondsRealtime(ISI);// ISI
                                }
                            }
                        }
                        textGameobject.SetActive(false);
                        currentInstructions.sprite = breakInstructions;
                        instructions.SetActive(true);
                        if (i<4)
                        {
                        while (!spacePress)
                        {
                            if (Input.GetMouseButtonDown(0))
                            {
                                spacePress=true;
                            }
                            yield return null;
                        }
                        spacePress=false;
                        instructions.SetActive(false);
                        textGameobject.SetActive(true);
                        }
                    }
                instructions.SetActive(true);
                currentInstructions.sprite = endScreen;
                textGameobject.SetActive(false);
                yield break;
            }

    private void ResetParameters()
    {
        trialTypeList = arr1.Concat(arr2).Concat(arr3).ToArray();
        reshuffle(trialTypeList);
        trialTypeList = zeroArr.Concat(trialTypeList).ToArray();
    }
}
