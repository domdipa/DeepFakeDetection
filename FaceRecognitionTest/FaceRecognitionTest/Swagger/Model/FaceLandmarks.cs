/*
 * FastAPI
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * The version of the OpenAPI document: 0.1.0
 * Generated by: https://github.com/openapitools/openapi-generator.git
 */


using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.IO;
using System.Runtime.Serialization;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System.ComponentModel.DataAnnotations;
using OpenAPIDateConverter = Org.OpenAPITools.Client.OpenAPIDateConverter;

namespace Org.OpenAPITools.Model
{
    /// <summary>
    /// FaceLandmarks
    /// </summary>
    [DataContract(Name = "FaceLandmarks")]
    public partial class FaceLandmarks : IValidatableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FaceLandmarks" /> class.
        /// </summary>
        [JsonConstructorAttribute]
        protected FaceLandmarks() { }
        /// <summary>
        /// Initializes a new instance of the <see cref="FaceLandmarks" /> class.
        /// </summary>
        /// <param name="imageShape">imageShape (required).</param>
        /// <param name="leftEye">leftEye (required).</param>
        /// <param name="rightEye">rightEye (required).</param>
        /// <param name="mouth">mouth (required).</param>
        /// <param name="leftIris">leftIris (required).</param>
        /// <param name="rightIris">rightIris (required).</param>
        /// <param name="noseTip">noseTip (required).</param>
        /// <param name="chin">chin (required).</param>
        /// <param name="leftCheek">leftCheek (required).</param>
        /// <param name="rightCheek">rightCheek (required).</param>
        public FaceLandmarks(string imageShape = default(string), List<LandmarkPoint> leftEye = default(List<LandmarkPoint>), List<LandmarkPoint> rightEye = default(List<LandmarkPoint>), List<LandmarkPoint> mouth = default(List<LandmarkPoint>), List<LandmarkPoint> leftIris = default(List<LandmarkPoint>), List<LandmarkPoint> rightIris = default(List<LandmarkPoint>), List<LandmarkPoint> noseTip = default(List<LandmarkPoint>), List<LandmarkPoint> chin = default(List<LandmarkPoint>), List<LandmarkPoint> leftCheek = default(List<LandmarkPoint>), List<LandmarkPoint> rightCheek = default(List<LandmarkPoint>))
        {
            // to ensure "imageShape" is required (not null)
            if (imageShape == null)
            {
                throw new ArgumentNullException("imageShape is a required property for FaceLandmarks and cannot be null");
            }
            this.ImageShape = imageShape;
            // to ensure "leftEye" is required (not null)
            if (leftEye == null)
            {
                throw new ArgumentNullException("leftEye is a required property for FaceLandmarks and cannot be null");
            }
            this.LeftEye = leftEye;
            // to ensure "rightEye" is required (not null)
            if (rightEye == null)
            {
                throw new ArgumentNullException("rightEye is a required property for FaceLandmarks and cannot be null");
            }
            this.RightEye = rightEye;
            // to ensure "mouth" is required (not null)
            if (mouth == null)
            {
                throw new ArgumentNullException("mouth is a required property for FaceLandmarks and cannot be null");
            }
            this.Mouth = mouth;
            // to ensure "leftIris" is required (not null)
            if (leftIris == null)
            {
                throw new ArgumentNullException("leftIris is a required property for FaceLandmarks and cannot be null");
            }
            this.LeftIris = leftIris;
            // to ensure "rightIris" is required (not null)
            if (rightIris == null)
            {
                throw new ArgumentNullException("rightIris is a required property for FaceLandmarks and cannot be null");
            }
            this.RightIris = rightIris;
            // to ensure "noseTip" is required (not null)
            if (noseTip == null)
            {
                throw new ArgumentNullException("noseTip is a required property for FaceLandmarks and cannot be null");
            }
            this.NoseTip = noseTip;
            // to ensure "chin" is required (not null)
            if (chin == null)
            {
                throw new ArgumentNullException("chin is a required property for FaceLandmarks and cannot be null");
            }
            this.Chin = chin;
            // to ensure "leftCheek" is required (not null)
            if (leftCheek == null)
            {
                throw new ArgumentNullException("leftCheek is a required property for FaceLandmarks and cannot be null");
            }
            this.LeftCheek = leftCheek;
            // to ensure "rightCheek" is required (not null)
            if (rightCheek == null)
            {
                throw new ArgumentNullException("rightCheek is a required property for FaceLandmarks and cannot be null");
            }
            this.RightCheek = rightCheek;
        }

        /// <summary>
        /// Gets or Sets ImageShape
        /// </summary>
        [DataMember(Name = "image_shape", IsRequired = true, EmitDefaultValue = true)]
        public string ImageShape { get; set; }

        /// <summary>
        /// Gets or Sets LeftEye
        /// </summary>
        [DataMember(Name = "left_eye", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> LeftEye { get; set; }

        /// <summary>
        /// Gets or Sets RightEye
        /// </summary>
        [DataMember(Name = "right_eye", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> RightEye { get; set; }

        /// <summary>
        /// Gets or Sets Mouth
        /// </summary>
        [DataMember(Name = "mouth", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> Mouth { get; set; }

        /// <summary>
        /// Gets or Sets LeftIris
        /// </summary>
        [DataMember(Name = "left_iris", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> LeftIris { get; set; }

        /// <summary>
        /// Gets or Sets RightIris
        /// </summary>
        [DataMember(Name = "right_iris", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> RightIris { get; set; }

        /// <summary>
        /// Gets or Sets NoseTip
        /// </summary>
        [DataMember(Name = "nose_tip", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> NoseTip { get; set; }

        /// <summary>
        /// Gets or Sets Chin
        /// </summary>
        [DataMember(Name = "chin", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> Chin { get; set; }

        /// <summary>
        /// Gets or Sets LeftCheek
        /// </summary>
        [DataMember(Name = "left_cheek", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> LeftCheek { get; set; }

        /// <summary>
        /// Gets or Sets RightCheek
        /// </summary>
        [DataMember(Name = "right_cheek", IsRequired = true, EmitDefaultValue = true)]
        public List<LandmarkPoint> RightCheek { get; set; }

        /// <summary>
        /// Returns the string presentation of the object
        /// </summary>
        /// <returns>String presentation of the object</returns>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("class FaceLandmarks {\n");
            sb.Append("  ImageShape: ").Append(ImageShape).Append("\n");
            sb.Append("  LeftEye: ").Append(LeftEye).Append("\n");
            sb.Append("  RightEye: ").Append(RightEye).Append("\n");
            sb.Append("  Mouth: ").Append(Mouth).Append("\n");
            sb.Append("  LeftIris: ").Append(LeftIris).Append("\n");
            sb.Append("  RightIris: ").Append(RightIris).Append("\n");
            sb.Append("  NoseTip: ").Append(NoseTip).Append("\n");
            sb.Append("  Chin: ").Append(Chin).Append("\n");
            sb.Append("  LeftCheek: ").Append(LeftCheek).Append("\n");
            sb.Append("  RightCheek: ").Append(RightCheek).Append("\n");
            sb.Append("}\n");
            return sb.ToString();
        }

        /// <summary>
        /// Returns the JSON string presentation of the object
        /// </summary>
        /// <returns>JSON string presentation of the object</returns>
        public virtual string ToJson()
        {
            return Newtonsoft.Json.JsonConvert.SerializeObject(this, Newtonsoft.Json.Formatting.Indented);
        }

        /// <summary>
        /// To validate all properties of the instance
        /// </summary>
        /// <param name="validationContext">Validation context</param>
        /// <returns>Validation Result</returns>
        IEnumerable<ValidationResult> IValidatableObject.Validate(ValidationContext validationContext)
        {
            yield break;
        }
    }

}
