# 🚀 Deployment Quick Start

Your project is ready to deploy! Choose your platform:

---

## **Option 1: Render.com (Recommended - Easiest)**

### Steps:
1. Go to https://render.com/
2. Sign in with GitHub
3. Click **"New +"** → **"Web Service"**
4. Select your repo: `rahiatkiamona/MLSD-PROJECT`
5. Render auto-detects `render.yaml` and `Dockerfile`
6. Click **Deploy**
7. Wait ~2 minutes for build completion
8. Your app will be live at: `https://<your-service-name>.onrender.com`

**Cost**: Free tier available (with limitations)

---

## **Option 2: Railway.app**

### Steps:
1. Go to https://railway.app/
2. Sign in with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repo
5. Railway detects Docker automatically
6. Set **PORT** environment variable to `8000`
7. Deploy
8. Live at: `https://<project-name>.up.railway.app`

**Cost**: $5/month free credits

---

## **Option 3: Fly.io**

### Steps:
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `flyctl auth signup` (create account)
3. In project folder: `flyctl launch`
4. Follow prompts, accept defaults
5. Deploy: `flyctl deploy`
6. Live at: `https://<app-name>.fly.dev`

**Cost**: Free tier available

---

## **Testing Deployed System**

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-deployed-url/health

# Available versions
curl https://your-deployed-url/api/versions

# Make a prediction (JSON)
curl -X POST https://your-deployed-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 2,
    "Administrative_Duration": 35.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 28,
    "ProductRelated_Duration": 980.0,
    "BounceRates": 0.01,
    "ExitRates": 0.04,
    "PageValues": 20.0,
    "SpecialDay": 0.0,
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": 0,
    "Month": "Nov",
    "threshold": 0.5
  }'
```

---

## **Production Checklist**

- ✅ Code pushed to GitHub
- ✅ Dockerfile created
- ✅ Requirements.txt with all dependencies
- ✅ render.yaml for Render deployment
- ✅ All artifacts (v1, v2) included
- ✅ API endpoints tested locally

**You're ready to deploy!** 🎉
