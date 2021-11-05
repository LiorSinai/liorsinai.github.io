using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    public Transform from;
    public Transform to;
    int screenshots_count = 0;
    float speed = 0.01f;
    float screenCaptureInterval = 0.1f;
    string output_name = "Slerp";
    float startTime;
    
    // Start is called before the first frame update
    void Start()
    {
        from = transform;
        startTime = Time.time;

        Debug.Log("q_start = " + from.rotation);
        Debug.Log("q_end =   " + to.rotation);
    }


    // Update is called once per frame
    void Update()
    {
        var t = Time.time * speed;
        transform.rotation = Quaternion.Slerp(from.rotation, to.rotation, t);
        Debug.Log("q = " + transform.rotation);
        CaptureSequence();
    }


    void CaptureSequence()
    {
        var currentTime = Time.time;
        if (currentTime - startTime > screenCaptureInterval)
        {
            startTime = currentTime;
            string filename = output_name + "_" + screenshots_count + ".png";
            screenshots_count += 1;
            ScreenCapture.CaptureScreenshot(filename);
        }
    }
}
