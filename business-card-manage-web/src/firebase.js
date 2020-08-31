import firebase from "firebase/app";
import "firebase/auth";
import "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAm6WyaGAHOI1YDfiU4aVm6Y2WKUk8QbQ4",
  authDomain: "business-card-manage-web.firebaseapp.com",
  databaseURL: "https://business-card-manage-web.firebaseio.com",
  projectId: "business-card-manage-web",
  storageBucket: "business-card-manage-web.appspot.com",
  messagingSenderId: "863860612789",
  appId: "1:863860612789:web:e62cfc737bc469c7cce8fc",
  measurementId: "G-3E9MY06B4W"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
export const auth = firebase.auth();
export const firestore = firebase.firestore();

const provider = new firebase.auth.GoogleAuthProvider();
export const signInWithGoogle = () => {
  auth.signInWithPopup(provider);
};

///////
const userRef = firestore.doc(`users/${user.uid}`);
const snapshot = await userRef.get();

export const generateUserDocument = async(user, additionalData) => {

};