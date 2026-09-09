using UnityEngine;

public class ToggleCameraClippingPlane : MonoBehaviour
{
    [Header("Cameras to Adjust")]
    [SerializeField] private Camera firstCamera;
    [SerializeField] private Camera secondCamera;
    
    [Header("Clipping Plane Values")]
    [SerializeField] private float nearClipValueA = 0.1f;
    [SerializeField] private float nearClipValueB = 0.35f;

    [Header("RemoteCameraWindow")] public GameObject remoteCameraWindow;

    private void Awake()
    {
        ApplySmallView();
    }

    private void OnEnable()
    {
        ApplySmallView();
    }

    private void ApplySmallView()
    {
        if (firstCamera != null)
        {
            firstCamera.nearClipPlane = nearClipValueB;
        }

        if (secondCamera != null)
        {
            secondCamera.nearClipPlane = nearClipValueB;
        }
    }
}
